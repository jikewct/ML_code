import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from accelerate import Accelerator

from network import *
from utils import monitor

from . import model_utils
from .ema import EMA
from .noise_schedule import *


class BaseModel(ABC):

    def __init__(self, config):
        self.init_parameters(config)

        self.init_coefficient(config)
        self.init_network(config)
        self.init_debug_info(config)

    def init_parameters(self, config):
        self.step = 0
        self.device = config.device
        self.img_size = config.data.img_size
        self.img_channels = config.data.img_channels
        self.num_classes = config.data.num_classes
        self.loss_type = config.model.loss_type
        self.log_to_wandb = config.training.log_to_wandb
        self.mode = model_utils.ModeEnum.TRAIN
        self.sampling_method = config.sampling.method
        self.sampling_config = config.sampling[self.sampling_method]

    def init_debug_info(self, config):
        self.sampling_enable_debug = config.sampling.enable_debug
        self.sampling_log_freq = config.sampling.log_freq
        self.debug_groups = config.training.debug_groups
        self.enable_debug = config.training.enable_debug
        self.enable_lora = config.model.enable_lora
        if self.enable_lora:
            self.lora_config = config.model.lora
        self.load_ckpt_strict = config.model.load_ckpt_strict
        self.debug_sampling_save_path = config.test.save_path + "/debug_sampling"
        self.loss_group_metric = monitor.HistogramMeter("loss")
        self.mmoe_weight1_metric = monitor.HistogramMeter("mmoe_w1")
        self.mmoe_weight2_metric = monitor.HistogramMeter("mmoe_w2")
        self.debug_sampling_step = monitor.RunningAverageMeter(0)

    def reset_debug_metrics(self):
        self.loss_group_metric.reset()
        self.mmoe_weight1_metric.reset()
        self.mmoe_weight2_metric.reset()

    def init_network(self, config):
        self.network = net_factory.create_network(config).to(config.device)
        self.enable_ema = config.model.enable_ema
        if self.enable_ema:
            self.init_ema(config)
        self.conditional = config.model.conditional
        if self.conditional:
            self.condition_type = config.model.condition.condition_type
            if self.condition_type == "text":
                self.init_cond_embedder(config)

    def init_cond_embedder(self, config):
        self.cond_embedder = net_factory.create_network(config.model.condition, "cond_embedder_name").to(config.device)
        # self.clip = clip.FrozenCLIPEmbedder(config.model.clip.pretrained_path).to(config.device)

    def init_ema(self, config):
        ema_config = config.model.ema
        self.ema = EMA(ema_config.ema_decay)
        self.ema_decay = ema_config.ema_decay
        self.ema_start = ema_config.ema_start
        self.ema_update_rate = ema_config.ema_update_rate
        self.ema_network = deepcopy(self.network)

    def init_sampler(self, config):
        from models.sample import sample_factory

        self.sampler = sample_factory.create_sampler(self, config)
        logging.info(f"sampler info:{self.sampler.states()}")

    def accelerator_prepare(self, accelerator: Accelerator):
        self.network = accelerator.prepare(self.network)
        if self.enable_ema:
            self.ema_network = accelerator.prepare(self.ema_network)

    def init_coefficient(self, config):
        self.ns = self.init_schedule(config)

    @abstractmethod
    def init_schedule(self, config) -> BaseNoiseSchedule:
        pass

    def parameters(self):
        return self.network.parameters()

    # def save_checkpoint(self, filepath):
    #     torch.save(self.network.state_dict(), filepath)
    #     if self.enable_ema:
    #         ema_filepath = filepath + "-ema"
    #         torch.save(self.ema_network.state_dict(), ema_filepath)

    def load_checkpoint(self, filepath):
        state_dict = torch.load(filepath)
        strict = True
        if self.load_ckpt_strict:
            strict = True
        elif self.enable_lora:
            strict = False
        self.network.load_state_dict(state_dict, strict)
        if self.enable_ema:
            self.ema_network.load_state_dict(state_dict, strict)
        if self.enable_lora:
            layer_utils.mark_only_lora_as_trainable(self.network, bias=self.lora_config.lora_bias_trainable)
            for name, p in list(self.network.named_parameters()):
                if p.requires_grad:
                    logging.debug(f"parameter: {name} grad:{p.requires_grad}, shape:{p.shape}")

    def load_state(self, state):
        self.network.load_state_dict(state["network"])
        if self.enable_ema:
            self.ema_network.load_state_dict(state["ema_network"])

    def state_dict(self):
        state = dict()
        state["network"] = self.network.state_dict()
        if self.enable_ema:
            state["ema_network"] = self.ema_network.state_dict()
        return state

    def set_train_mode(self):
        self.mode = model_utils.ModeEnum.TRAIN
        self.network.train()
        if self.enable_ema:
            self.ema_network.train()

    def set_eval_mode(self):
        self.mode = model_utils.ModeEnum.EVAL
        self.network.eval()
        if self.enable_ema:
            self.ema_network.eval()

    def set_sampling_mode(self):
        self.set_eval_mode()
        self.mode = model_utils.ModeEnum.SAMPLING

    def set_test_mode(self):
        self.set_eval_mode()
        self.mode = model_utils.ModeEnum.TEST

    def get_network(self):
        return self.network

    def update_ema(self):
        if not self.enable_ema:
            return
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_network.load_state_dict(self.network.state_dict())
            else:
                self.ema.update_model_average(self.ema_network, self.network)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def predict(self, x, t, y=None, use_ema=False):
        _x, _t, _y, _use_ema = self.before_predict(x, t, y, use_ema)
        # logging.info(f"t_shape:{_t.shape}, t_max:{_t.max()}, t_min:{_t.min()}")
        if _use_ema and self.enable_ema:
            output = self.ema_network(_x, _t, _y)
        else:
            output = self.network(_x, _t, _y)
        if isinstance(output, tuple):
            preds, extra_info = output[0], output[1]
        else:
            preds, extra_info = output, {}
        preds, extra_info = self.after_predict(x, t, y, use_ema, preds, extra_info)
        return preds, extra_info

    def before_predict(self, x, t, y, use_ema):
        return x, t, y, use_ema

    def after_predict(self, x, t, y, user_ema, preds, extra_info):
        if self.mode == model_utils.ModeEnum.SAMPLING:
            self.debug_sampling(x, t, preds, extra_info)
        return preds, extra_info

    def sampling_predict(self, x, t, y, use_ema, uncond_y, guidance_scale):
        if uncond_y is None or guidance_scale <= 1e-4:
            preds, extra_info = self.predict(x, t, y, use_ema=use_ema)
        else:
            preds, extra_info = self.cfg_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        return preds, extra_info

    def cfg_predict(self, x, t, y, use_ema, uncond_y, guidance_scale):
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        y_in = torch.cat([uncond_y, y])
        speed_vf_tmp, extra_info = self.predict(x_in, t_in, y_in, use_ema=use_ema)
        speed_vf_uncond, speed_vf_cond = speed_vf_tmp.chunk(2)
        speed_vf = speed_vf_cond + guidance_scale * (speed_vf_cond - speed_vf_uncond)
        return speed_vf, extra_info

    @abstractmethod
    def forward(self, x, y=None):
        pass

    def __call__(self, x, y=None):
        preds, targets, info = self.forward(x, y)
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)
        if self.loss_type == "l2":
            losses = self.l2_loss(preds, targets)
        if self.enable_debug:
            self.aggregate_info_by_group(losses, info)
        return losses.mean()

    def l2_loss(self, preds, targets):
        losses = 1 / 2 * ((preds - targets) ** 2).sum(dim=1)
        return losses

    def encode_text_condition(self, text_condition):
        return self.cond_embedder.encode(text_condition)

    @abstractmethod
    @torch.no_grad()
    def sample(self, batch_size, y=None, use_ema=True, uncond_y=None, guidance_scale=0.0) -> torch.Tensor:
        pass

    @abstractmethod
    def prior_sampling(self, batch_size, device):
        pass

    @abstractmethod
    def marginal_std(self, t):
        pass

    @property
    def T(self):
        return self.ns.T

    @property
    def EPS(self):
        return self.ns.EPS

    @property
    def N(self):
        return self.ns.N

    def get_groups(self, t):
        interval = self.T / self.debug_groups
        groups = torch.div(t, interval, rounding_mode="trunc")
        # groups = (t // interval).long()
        return groups

    @torch.no_grad()
    def aggregate_info_by_group(self, loss, info):
        t = info["t"]
        groups = self.get_groups(t)
        if "time_gate_softmax" in info:
            w1 = info["time_gate_softmax"][:, 0]
            w2 = info["time_gate_softmax"][:, 1]
            self.mmoe_weight1_metric.add_metric_with_group(w1, groups)
            self.mmoe_weight2_metric.add_metric_with_group(w2, groups)
        self.loss_group_metric.add_metric_with_group(loss, groups)

    def get_debug_info(self):
        debug_info = dict()
        debug_info.update(self.loss_group_metric.get_avg_metric_dict())
        debug_info.update(self.mmoe_weight1_metric.get_avg_metric_dict_without_cnt())
        debug_info.update(self.mmoe_weight2_metric.get_avg_metric_dict_without_cnt())
        self.reset_debug_metrics()
        return debug_info

    def debug_sampling(self, x, t, preds, extra_info):
        self.debug_sampling_step.inc()
        if (
            not self.sampling_enable_debug
            or not self.log_to_wandb
            or self.mode != model_utils.ModeEnum.SAMPLING
            or self.debug_sampling_step.val % self.sampling_log_freq != 0
        ):
            return
        assert x.shape[0] > 0
        x_0, preds_0, sigma = x[0], preds[0], self.marginal_std(t)[0]
        expected_norm = self.cal_expected_norm(sigma)
        x_mean, x_max, x_min = x_0.mean(), x_0.max(), x_0.min()
        preds_0_norm = torch.norm(preds_0)
        sampling_info = {
            "sigma": sigma,
            "x_mean": x_mean,
            "x_max": x_max,
            "x_min": x_min,
            "preds_norm": preds_0_norm,
            "expected_norm": expected_norm,
            "sampling_step": self.debug_sampling_step.val,
        }
        if "log" in extra_info.keys():
            sampling_info.update(extra_info["log"])

        cvt_x = torch.clone(x_0)
        cvt_x = ((cvt_x - x_min) / (x_max - x_min + 1e-6) * 255).permute(1, 2, 0).detach().cpu().numpy()
        cvt_x = np.squeeze(cvt_x.round().astype(np.uint8))

        if self.log_to_wandb:
            wandb.log(sampling_info)
            # if self.debug_sampling_step.val % (self.sampling_log_freq * 10) == 0:
            wandb.log(
                {
                    "debug_sampling": wandb.Image(cvt_x),
                    "sampling_step": self.debug_sampling_step.val,
                }
            )
        os.makedirs(self.debug_sampling_save_path, exist_ok=True)
        save_filename = "{}/{:08d}.png".format(self.debug_sampling_save_path, self.debug_sampling_step.val)
        imageio.imwrite(save_filename, cvt_x)
        logging.debug(sampling_info)

    def cal_expected_norm(self, sigma):
        return np.sqrt(self.img_size[0] * self.img_size[1] * self.img_channels) / sigma
