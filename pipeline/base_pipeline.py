import logging
import os
import time
from abc import ABC, abstractmethod

import imageio
import numpy as np
import pytorch_fid
import pytorch_fid.fid_score
import torch
import wandb
from accelerate import Accelerator
from torch import Tensor, optim

from datasets import datasetsHelper
from models import *
from utils import monitor


class BasePipeLine(ABC):

    def __init__(self, config):
        self.init_parameters(config)
        # self.init_data_loader()
        self.init_model()
        self.init_optimizer()
        self.load_checkpoint()

    def init_parameters(self, config):
        self.accelerator = Accelerator()
        config.device = self.accelerator.device
        logging.info(f"device:{config.device}")
        self.config = config
        self.device = self.config.device
        self.use_labels = self.config.model.use_labels
        self.sample_steps = self.config.sampling.sample_steps

    def init_model(self):
        self.model = model_factory.create_model(self.config)

    def init_data_loader(self):

        self.train_loader, self.test_loader = datasetsHelper.get_loader(
            self.config.data.dataset,
            root=self.config.data.root_path,
            batch_size=self.config.training.batch_size,
            img_size=self.config.data.img_size,
            img_class=self.config.data.img_class,
        )

    def init_optimizer(self):
        config = self.config
        parameters = self.model.parameters()
        if config.optim.optimizer == "Adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=config.optim.lr,
                weight_decay=config.optim.weight_decay,
                betas=(config.optim.beta1, 0.999),
                amsgrad=config.optim.amsgrad,
                eps=config.optim.eps,
            )
        elif config.optim.optimizer == "RMSProp":
            self.optimizer = optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == "SGD":
            self.optimizer = optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError("Optimizer {} not understood.".format(config.optim.optimizer))

    def accelerator_prepare(self):
        self.train_loader, self.test_loader = self.accelerator.prepare(self.train_loader, self.test_loader)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.model.accelerator_prepare(self.accelerator)

    def init_metrics(self):
        self.current_epoch = monitor.RunningAverageMeter(0)
        self.current_step = monitor.RunningAverageMeter(0)
        self.train_loss = monitor.RunningAverageMeter(0)
        self.eval_loss = monitor.RunningAverageMeter(0)
        self.log_time_meter = monitor.TimeMeter()
        self.eval_time_meter = monitor.TimeMeter()
        self.test_time_meter = monitor.TimeMeter()

    def reset_loss(self):
        self.train_loss.reset()
        self.eval_loss.reset()

    def load_checkpoint(self):
        model_checkpoint = self.config.training.model_checkpoint
        # optim_checkpoint = self.config.training.optim_checkpoint
        if model_checkpoint is None or model_checkpoint.rstrip() == "":
            return
        self.model.load_checkpoint(model_checkpoint)
        logging.info(f"load model checkpoint success! {model_checkpoint}")

    def save_checkpoint(self):
        filepath = f"{self.config.training.ckpt_dir}/{self.config.training.project_name}/{self.config.model.name}/{self.config.model.nn_name}-{self.config.data.dataset}-{self.current_step.val}-model"
        os.makedirs(os.path.split(filepath)[0], exist_ok=True)
        self.set_train_mode()  ## for lora model to split parameters
        self.model.save_checkpoint(filepath)
        logging.info(f"save model checkpoint success! {filepath}")

    def train_loop(self):
        self.before_train()
        # train epoch
        for _ in range(1, self.config.training.epochs):
            self.before_epoch()
            # for  train step
            for x, y in self.train_loader:
                # logging.info(f"x dtype:{x.dtype}, y dtype:{y.dtype}")
                self.before_step()
                train_step_loss = self.step_train(x, y)
                self.train_loss.update(train_step_loss.item())
                self.after_step()
            self.after_epoch()
        self.after_train()
        #### end train step

    def before_step(self):
        self.set_train_mode()
        if self.current_step.val == 0 or self.current_step.val % self.config.training.log_freq == 1:
            self.log_time_meter.start()

    def after_step(self):
        step = self.current_step.val
        if step % self.config.training.log_freq == 0:
            self.log_time_meter.stop()
            self.log_train_status()
        if step % self.config.training.eval_freq == 0:
            self.eval_loop()
        if step % self.config.training.snapshot_freq == 0:
            self.save_checkpoint()
        if step > self.config.fast_fid.begin_step and step % self.config.training.test_metric_freq == 0:
            self.test_metric_loop()
        self.current_step.inc()

    def before_eval(self):
        self.set_eval_mode()
        self.eval_time_meter.start()

    def after_eval(self):
        self.eval_time_meter.stop()
        self.log_eval_status()

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
        logging.info(f"fid score:{fid_score}, train steps: {self.current_step.val}, image path:{save_path}")
        if self.config.training.log_to_wandb:
            wandb.log(
                {
                    "fid_score": fid_score,
                    "train_step": self.current_step.val / self.config.training.log_freq,
                }
            )

    def generate_samples(self, batch_size, sample_num, save_path, desc=""):
        self.test_time_meter.start()
        logging.info(f"==================begin {desc} generate samples =======================")
        self.before_eval()
        batch_size_list = np.concatenate(
            [
                np.asarray([batch_size] * (sample_num // batch_size), dtype=np.int64),
                [sample_num % batch_size],
            ]
        )
        batch_size_list = np.delete(batch_size_list, np.where(batch_size_list == 0))
        for iter, num in tqdm.tqdm(enumerate(batch_size_list), desc="Processing", total=len(batch_size_list)):
            test_samples = self.sample(num, steps=self.sample_steps)
            self.save_result(test_samples, iter * batch_size, save_path)
        self.test_time_meter.stop()
        logging.info("generate {} samples, time cost:{:.2f} seconds, save_path:{}".format(sample_num, self.test_time_meter.interval(), save_path))
        logging.info(f"==================end {desc} generate samples =======================")

    def log_eval_status(self):
        debug_info = self.model.get_debug_info()
        logging.debug(debug_info)
        logging.info(
            "====epoch:{:04d},step:{:08d}, eval_loss:{:.8f}, eval_cost:{:.2f} ====".format(
                self.current_epoch.val,
                self.current_step.val,
                self.eval_loss.avg,
                self.eval_time_meter.interval(),
            )
        )
        if self.config.training.log_to_wandb:
            eval_log = {
                "eval_loss": self.eval_loss.avg,
                "eval_cost": self.eval_time_meter.interval(),
                "samples": [wandb.Image(sample) for sample in self.eval_samples],
                "train_step": self.current_step.val / self.config.training.log_freq,
            }
            wandb.log(eval_log)

    def log_train_status(self):
        debug_info = self.model.get_debug_info()
        logging.debug(debug_info)
        logging.info(
            "epoch:{:04d},step:{:08d}, train_loss:{:.8f}, total_cost:{:.2f}, lr:{:.8f}".format(
                self.current_epoch.val,
                self.current_step.val,
                self.train_loss.avg,
                self.log_time_meter.interval(),
                self.get_lr(),
            )
        )
        if self.config.training.log_to_wandb:
            debug_info.update(
                {
                    "train_loss": self.train_loss.avg,
                    "lr": self.get_lr(),
                    "train_cost": self.log_time_meter.interval(),
                    "train_step": self.current_step.val / self.config.training.log_freq,
                }
            )
            wandb.log(debug_info)

        self.reset_loss()

    def get_model(self):
        return self.model.get_network()

    def set_train_mode(self):
        self.model.set_train_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()

    def set_sampling_mode(self):
        self.model.set_sampling_mode()

    def set_test_mode(self):
        self.model.set_test_mode()

    def step_train(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        if self.use_labels:
            loss = self.model(x, y)
        else:
            loss = self.model(x)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.model.update_ema()
        return loss

    def step_eval(self, x, y):
        x, y = x.to(self.device), y.to(self.device)

        if self.use_labels:
            loss = self.model(x, y)
        else:
            loss = self.model(x)
        return loss

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def eval_loop(self):
        with torch.no_grad():
            self.before_eval()
            for x, y in self.test_loader:
                test_step_loss = self.step_eval(x, y)
                self.eval_loss.update(test_step_loss.item())
            self.eval_samples = self.sample(5, steps=self.sample_steps)
            self.after_eval()

    def sample(self, num=10, steps=100) -> torch.Tensor:
        self.set_sampling_mode()
        if self.use_labels:
            y = torch.arange(self.config.data.num_classes, device=self.device)
            samples = self.model.sample(num, self.device, y, steps=steps)
        else:
            samples = self.model.sample(num, self.device, steps=steps)
        samples = ((samples + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        return samples

    def test_loop(self):
        self.init_metrics()
        sample_num = self.config.test.num_samples
        batch_size = self.config.test.batch_size
        save_path = f"{self.config.test.save_path}/{self.config.model.name}/{self.config.data.dataset}"
        self.generate_samples(batch_size, sample_num, save_path, "test")

    def before_train(self):
        self.init_data_loader()
        self.accelerator_prepare()
        self.init_metrics()

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        self.current_epoch.inc()

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
        self.init_metrics()
        self.generate_samples(1, 1, "./data/test/debug_sampling/", "debug_sampling")
