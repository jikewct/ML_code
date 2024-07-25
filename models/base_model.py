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

from network import ncsnv2, net_utils
from utils import monitor

from . import model_utils
from .ema import EMA


class BaseModel(ABC):

    def __init__(self, config):
        self.init_parameters(config)
        self.init_coefficient(config)
        self.init_network(config)
        self.init_debug_info(config)

    def init_parameters(self, config):
        self.step = 0
        self.img_size = config.data.img_size
        self.img_channels = config.data.img_channels
        self.num_classes = config.data.num_classes
        self.loss_type = config.model.loss_type
        self.log_to_wandb = config.training.log_to_wandb
        self.mode = model_utils.ModeEnum.TRAIN

    def init_debug_info(self, config):
        self.enable_debug_sampling = config.sampling.enable_debug
        self.sampling_log_freq = config.sampling.log_freq
        self.debug_groups = config.training.debug_groups
        self.enable_debug = config.training.enable_debug
        self.enable_lora = any(config.model.enable_lora)
        self.load_ckpt_strict = config.model.load_ckpt_strict
        self.lora_bias_trainable = config.model.lora_bias_trainable
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
        self.network = net_utils.create_network(config).to(config.device)
        self.enable_ema = config.model.enable_ema
        if self.enable_ema:
            self.init_ema(config)

    def init_ema(self, config):
        self.ema = EMA(config.model.ema_decay)
        self.ema_decay = config.model.ema_decay
        self.ema_start = config.model.ema_start
        self.ema_update_rate = config.model.ema_update_rate
        self.ema_network = deepcopy(self.network)

    def accelerator_prepare(self, accelerator: Accelerator):
        self.network = accelerator.prepare(self.network)
        if self.enable_ema:
            self.ema_network = accelerator.prepare(self.ema_network)

    @abstractmethod
    def init_coefficient(self, config):
        pass

    def parameters(self):
        return self.network.parameters()

    def save_checkpoint(self, filepath):
        torch.save(self.network.state_dict(), filepath)
        if self.enable_ema:
            ema_filepath = filepath + "-ema"
            torch.save(self.ema_network.state_dict(), ema_filepath)

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
            net_utils.mark_only_lora_as_trainable(self.network, bias=self.lora_bias_trainable)
            for name, p in list(self.network.named_parameters()):
                if p.requires_grad:
                    logging.debug(f"parameter: {name} grad:{p.requires_grad}, shape:{p.shape}")

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
        if use_ema and self.enable_ema:
            output, extra_info = self.ema_network(x, t, y)
        else:
            output, extra_info = self.network(x, t, y)
        return output, extra_info

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

    @abstractmethod
    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, steps=1000):
        pass

    @abstractmethod
    def prior_sampling(self, batch_size, device):
        pass

    @abstractmethod
    def marginal_std(self, t):
        pass

    @property
    def T(self):
        pass

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
            not self.enable_debug_sampling
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
        save_filename = "{}/{:08d}.jpg".format(self.debug_sampling_save_path, self.debug_sampling_step.val)
        imageio.imwrite(save_filename, cvt_x)
        logging.debug(sampling_info)

    def cal_expected_norm(self, sigma):
        return np.sqrt(self.img_size[0] * self.img_size[1] * self.img_channels) / sigma

    def to_flattened_numpy(self, x):
        """Flatten a torch tensor `x` and convert it to numpy."""
        return x.detach().cpu().numpy().reshape((-1,))

    def from_flattened_numpy(self, x, shape):
        """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
        return torch.from_numpy(x.reshape(shape))
