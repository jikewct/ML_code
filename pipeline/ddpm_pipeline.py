import logging
import os
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import tqdm
import wandb

from models import *
from models.ddpm import DDPM
from utils import monitor

from .base_pipeline import BasePipeLine

__all__ = ["DDPMPipeLine"]


class DDPMPipeLine(BasePipeLine):

    def __init__(self, config):
        super().__init__(config)

    def init_optimizer(self):
        super().init_optimizer()
        if self.config.optim.lr_schedule == "MultiStepLR":
            decay_num = self.config.optim.decay_num
            decay_epochs = (self.config.training.epochs * (1 - 1 / np.power(decay_num, np.arange(decay_num) + 1))).astype(np.uint)
            self.step_lr = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=decay_epochs.tolist(),
                gamma=self.config.optim.weight_decay,
            )

    def after_epoch(self):
        super().after_epoch()
        self.lr_step()

    def lr_step(self):
        if self.config.optim.lr_schedule == "MultiStepLR":
            self.step_lr.step()
