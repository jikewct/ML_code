import logging
import os
import time
import uuid
from abc import ABC, abstractmethod

import accelerate
import einops
import imageio
import numpy as np
import pytorch_fid
import pytorch_fid.fid_score
import torch
import tqdm
import wandb
from accelerate import Accelerator
from torch import Tensor, optim

from datasets import *
from lib import tensor_trans
from models import *
from optimizer import optimize
from utils import monitor


class BasePipeLine(ABC):

    def __init__(self, config):
        self.init_parameters(config)
        # self.init_data_loader()
        self.init_seed()
        self.init_model()
        self.init_optimizer()
        self.init_metrics()
        self.load_pipeline()

    def init_parameters(self, config):
        self.accelerator = Accelerator()
        config.device = self.accelerator.device
        # logging.info(f"device:{config.device}")
        self.config = config
        self.device = self.config.device
        self.conditional = self.config.model.conditional
        if self.conditional:
            self.sampling_conditions = self.config.sampling.sampling_conditions
            self.guidance_scale = self.config.sampling.guidance_scale
            self.condition_param = self.config.model.condition
            self.empty_latent = None
            if self.condition_param.empty_latent_path != "":
                self.empty_latent = torch.from_numpy(np.load(self.condition_param.empty_latent_path)).to(self.device)
                # logging.info(self.empty_latent.shape)

        self.uuid = str(uuid.uuid1())
        self.init_internal_parameters()

    def init_internal_parameters(self):
        conditinal = "cond" if self.conditional else "uncond"
        self.ckpt_path = f"{self.config.training.ckpt_dir}/{self.config.training.project_name}/{self.config.model.name}/{self.config.model.nn_name}/{self.config.data.dataset}/{self.config.data.img_size[0]}X{self.config.data.img_size[1]}/{conditinal}"
        os.makedirs(self.ckpt_path, exist_ok=True)

    def init_seed(self):
        accelerate.utils.set_seed(self.config.seed, device_specific=True)

    def init_model(self):
        self.model = model_factory.create_model(self.config)

    def init_optimizer(self):
        config = self.config
        parameters = self.model.parameters()
        self.optimizer = optimize.get_optimizer(parameters, **config.optim)
        self.lr_scheduler = optimize.get_lr_scheduler(self.optimizer, **config.lr_scheduler)

    def init_data_loader(self):
        # data_config = self.config.data
        # dataset_config = data_config[data_config.dataset] if hasattr(data_config, data_config.dataset) else dict()
        self.dataset = dataset_factory.create_dataset(self.config)
        logging.info(self.dataset.states())
        self.train_loader, self.test_loader = self.dataset.get_dataloader(self.config.training.batch_size)

    def init_metrics(self):
        self.current_train_epoch = 0
        self.current_train_step = 0
        self.train_loss = monitor.RunningAverageMeter(0)
        self.eval_loss = monitor.RunningAverageMeter(0)
        self.log_time_meter = monitor.TimeMeter()
        self.eval_time_meter = monitor.TimeMeter()
        self.test_time_meter = monitor.TimeMeter()

    def accelerator_prepare(self):
        self.train_loader, self.test_loader = self.accelerator.prepare(self.train_loader, self.test_loader)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.model.accelerator_prepare(self.accelerator)

    def reset_loss(self):
        self.train_loss.reset()
        self.eval_loss.reset()

    def load_pipeline(self):
        if self.config.training.resume:
            self.resume_state()
        else:
            self.load_checkpoint()

    def load_checkpoint(self):
        model_checkpoint = self.config.training.model_checkpoint
        # optim_checkpoint = self.config.training.optim_checkpoint
        if model_checkpoint is None or model_checkpoint.rstrip() == "":
            return
        self.model.load_checkpoint(model_checkpoint)
        logging.info(f"load model checkpoint success! {model_checkpoint}")

    # def save_checkpoint(self):
    #     filepath = f"{self.config.training.ckpt_dir}/{self.config.training.project_name}/{self.config.model.name}/{self.config.model.nn_name}-{self.config.data.dataset}-{self.current_step}-model"
    #     os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    #     self.set_train_mode()  ## for lora model to split parameters
    #     self.model.save_checkpoint(filepath)
    #     logging.info(f"save model checkpoint success! {filepath}")

    def save_state(self):
        self.set_train_mode()  ## for lora model to split parameters
        ## save step
        state = self.state_dict()
        torch.save(state, os.path.join(self.ckpt_path, f"{self.current_train_step}.ckpt"))
        ## save network file for load checkpoint  or import easier
        torch.save(state["network"], os.path.join(self.ckpt_path, f"{self.current_train_step}-network.pth"))
        logging.info(f"save state success! path: {self.ckpt_path}, step: {self.current_train_step}")

    def load_state(self, step, path=None):
        if path is None:
            path = self.ckpt_path
        state = torch.load(os.path.join(path, f"{step}.ckpt"))
        self.current_train_epoch, self.current_train_step, self.uuid = state["pipeline"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.model.load_state(state)
        logging.info(f"load state success! path: {path}, step: {step}")

    def resume_state(self):
        path = self.config.training.resume_path
        step = self.config.training.resume_step
        if path is None or path.rstrip() == "":
            path = self.ckpt_path
        if not os.path.exists(path):
            logging.warn(f"resume path not found :{path}")
            return
        if step is None or step < 0:
            steps = list(filter(lambda x: ".ckpt" in x, os.listdir(path)))
            if not steps:
                logging.warn(f"resume step file not found from path:{path}")
                return
            steps = map(lambda x: int(x.split(".")[0]), steps)
            step = max(steps)
        logging.info(f"resume from steps:{step}")
        self.load_state(step, path)

    def state_dict(self):
        state = dict()
        state["pipeline"] = (self.current_train_epoch, self.current_train_step, self.uuid)
        state["optimizer"] = self.optimizer.state_dict()
        state["lr_scheduler"] = self.lr_scheduler.state_dict()
        state.update(self.model.state_dict())
        return state

    def train_loop(self):
        self.before_train()
        # train epoch
        for _ in range(1, self.config.training.epochs):
            self.before_train_epoch()
            # for  train step
            for x, y in self.train_loader:
                # logging.info(f"x shape:{x.shape},x dtype:{x.dtype},y shape:{y.shape} ,y dtype:{y.dtype}")
                x, y = self.before_train_step(x, y)
                train_step_loss = self.train_step(x, y)
                self.train_loss.update(train_step_loss.item())
                self.after_train_step()
            self.after_train_epoch()
        self.after_train()
        #### end train step

    def before_train(self):
        self.init_data_loader()
        self.accelerator_prepare()
        # self.init_metrics()
        self.watch_model()

    def before_train_epoch(self):
        pass

    def before_train_step(self, x, y):
        self.set_train_mode()
        self.log_time_meter.start()
        x, y = x.to(self.device), y.to(self.device)
        if self.conditional:
            y = self.condition_transform(y)
        return x, y

    def train_step(self, x, y):

        if self.conditional:
            loss = self.model(x, y)
        else:
            loss = self.model(x)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.update_ema()
        return loss

    def condition_transform(self, y):
        if self.condition_param.cfg:
            replaced_data = self.empty_latent if self.condition_param.condition_type == "text" else torch.zeros_like(y)
            y = tensor_trans.replace_sample_by_cond(y, replaced_data, self.condition_param.p_cond)
        return y

    def after_train_step(self):
        step = self.current_train_step
        if step % self.config.training.log_freq == 0:
            self.log_time_meter.stop()
            self.log_train_status()
        if step % self.config.training.eval_freq == 0:
            torch.cuda.empty_cache()
            self.eval_loop()
            torch.cuda.empty_cache()
        if step % self.config.training.snapshot_freq == 0:
            torch.cuda.empty_cache()
            # self.save_checkpoint()
            self.save_state()
            torch.cuda.empty_cache()
        if step > self.config.fast_fid.begin_step and step % self.config.training.test_metric_freq == 0:
            torch.cuda.empty_cache()
            self.test_metric_loop()
            torch.cuda.empty_cache()
        self.current_train_step += 1

    def after_train_epoch(self):
        self.current_train_epoch += 1

    def after_train(self):
        pass

    def eval_loop(self):
        with torch.no_grad():
            self.before_eval()
            for x, y in self.test_loader:
                x, y = self.before_eval_step(x, y)
                test_step_loss = self.eval_step(x, y)
                self.eval_loss.update(test_step_loss.item())
                self.after_eval_step()
            self.eval_samples = self.sample(5)
            self.after_eval()

    def before_eval(self):
        self.set_eval_mode()
        self.eval_time_meter.start()

    def before_eval_step(self, x, y):
        return x, y

    def eval_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        if self.conditional:
            loss = self.model(x, y)
        else:
            loss = self.model(x)
        return loss

    def after_eval_step(self):
        pass

    def after_eval(self):
        self.eval_time_meter.stop()
        self.log_eval_status()

    def log_eval_status(self):
        log_info = self.model.get_debug_info()
        logging.debug(log_info)
        log_info.update(monitor.cuda_memory(self.device))
        log_info.update(
            {
                "epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
                "eval_loss": self.eval_loss.avg,
                "eval_cost": self.eval_time_meter.interval(),
                "samples": [wandb.Image(sample) for sample in self.eval_samples],
            }
        )
        logging.info(
            "====epoch:{:04d},train_step:{:08d}, eval_loss:{:.8f}, eval_cost:{:.2f}, m_mr:{:.2f}, m_a:{:.2f}, m_ma:{:.2f}, m_r:{:.2f} ====".format(
                log_info["epoch"],
                log_info["train_step"],
                log_info["eval_loss"],
                log_info["eval_cost"],
                log_info["m_max_reserved"],
                log_info["m_allocated"],
                log_info["m_max_allocated"],
                log_info["m_reserved"],
            )
        )
        if self.config.training.log_to_wandb:
            wandb.log(log_info)

    def log_train_status(self):
        log_info = self.model.get_debug_info()
        logging.debug(log_info)
        log_info.update(monitor.cuda_memory(self.device))
        log_info.update(
            {
                "epoch": self.current_train_epoch,
                "train_step": self.current_train_step,
                "train_loss": self.train_loss.avg,
                "lr": self.get_lr(),
                "train_cost": self.log_time_meter.interval(),
            }
        )

        logging.info(
            "epoch:{:04d},train_step:{:08d}, train_loss:{:.8f}, train_cost:{:.2f}, lr:{:.8f}, m_mr:{:.2f}, m_a:{:.2f},  m_ma:{:.2f},m_r:{:.2f}".format(
                log_info["epoch"],
                log_info["train_step"],
                log_info["train_loss"],
                log_info["train_cost"],
                log_info["lr"],
                log_info["m_max_reserved"],
                log_info["m_allocated"],
                log_info["m_max_allocated"],
                log_info["m_reserved"],
            )
        )
        if self.config.training.log_to_wandb:
            wandb.log(log_info)

        self.reset_loss()

    def test_metric_loop(self):
        self.test_fid_score()

    def test_fid_score(self):
        sample_num = self.config.fast_fid.num_samples
        batch_size = self.config.fast_fid.batch_size
        ds_state_file = self.config.fast_fid.ds_state_file
        save_path = "{}/fast_fid/{}/".format(self.config.fast_fid.save_path, int(time.time()))
        self.generate_samples(batch_size, sample_num, save_path, "test fid")
        paths = [ds_state_file, save_path]
        fid_score = pytorch_fid.fid_score.calculate_fid_given_paths(paths, 50, self.device, 2048)
        logging.info(f"fid score:{fid_score}, train steps: {self.current_train_step}, image path:{save_path}")
        if self.config.training.log_to_wandb:
            wandb.log(
                {
                    "fid_score": fid_score,
                    "train_step": self.current_train_step / self.config.training.log_freq,
                }
            )

    def generate_samples(self, batch_size, sample_num, save_path, desc=""):
        self.test_time_meter.start()
        logging.info(f"==================begin {desc} generate samples =======================")
        self.before_eval()
        batch_size_list = np.concatenate([np.asarray([batch_size] * (sample_num // batch_size), dtype=np.int64), [sample_num % batch_size]])
        batch_size_list = np.delete(batch_size_list, np.where(batch_size_list == 0))
        for iter, num in tqdm.tqdm(enumerate(batch_size_list), desc="Processing", total=len(batch_size_list)):
            test_samples = self.sample(num)
            self.save_result(test_samples, iter * batch_size, save_path)
        self.test_time_meter.stop()
        logging.info("generate {} samples, time cost:{:.2f} seconds, save_path:{}".format(sample_num, self.test_time_meter.interval(), save_path))
        logging.info(f"==================end {desc} generate samples =======================")

    def watch_model(self):
        if self.config.training.log_to_wandb:
            wandb.watch(self.model.get_network())

    def set_train_mode(self):
        self.model.set_train_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()

    def set_sampling_mode(self):
        self.model.set_sampling_mode()

    def set_test_mode(self):
        self.model.set_test_mode()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def generate_sample_condition(self, num):
        # logging.info(self.sampling_conditions)
        # logging.info(self.condition_param)
        if self.sampling_conditions is None:
            y = torch.randint(0, self.config.data.num_classes, [num], device=self.device)
            if self.condition_param.cfg:
                uncond_y = torch.zeros_like(y)
        else:
            label_length = len(self.sampling_conditions)
            y = self.sampling_conditions * (num // label_length + 1)
            if self.condition_param.condition_type == "class":
                y = torch.as_tensor(y[:num], dtype=torch.int, device=self.device)
                uncond_y = torch.zeros_like(y)
            else:
                y = self.model.encode_text_condition(y[:num])
                uncond_y = einops.repeat(self.empty_latent, "1 L D -> B L D", B=num)
        # logging.info(y.shape, uncond_y.shape)
        return y, uncond_y

    def sample(self, num=10):
        self.set_sampling_mode()
        if self.conditional:
            y, uncond_y = self.generate_sample_condition(num)
            # logging.info(f"y shape:{y.shape}")
            samples = self.model.sample(num, y=y, uncond_y=uncond_y, guidance_scale=self.guidance_scale)
            # if self.condition_param.cfg and self.sample_scale >= 1e-4:
            #     uncond_samples = self.model.sample(num, torch.zeros_like(y))
            #     samples = samples + self.sample_scale * (samples - uncond_samples)
        else:
            samples = self.model.sample(num)
        ## model sample output : latent feature or img
        ## img : shape( B C H W) , range(-1, 1)
        samples = self.unpreprocess_after_sample(samples)
        samples = ((samples + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return samples

    def unpreprocess_after_sample(self, samples):
        return samples

    def test_loop(self):
        # self.init_metrics()
        sample_num = self.config.test.num_samples
        batch_size = self.config.test.batch_size
        save_path = f"{self.config.test.save_path}/{self.config.model.name}/{self.config.data.dataset}"
        self.generate_samples(batch_size, sample_num, save_path, "test")

    def save_result(self, results, iter, save_path):
        results = (results * 255).round().astype(np.uint8)
        os.makedirs(save_path, exist_ok=True)
        index = 0
        for sample in results:
            total_index = iter + index
            save_filename = "{}/{:05d}.jpg".format(save_path, total_index)
            imageio.imwrite(save_filename, np.squeeze(sample))
            index += 1

    def debug_sampling(self):
        # self.init_metrics()
        self.generate_samples(1, 1, "./data/test/debug_sampling/", "debug_sampling")
