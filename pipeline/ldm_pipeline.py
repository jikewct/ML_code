import logging
import os
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import tqdm
import wandb

from models import *

from .base_pipeline import BasePipeLine

__all__ = ["LDMPipeLine"]


class LDMPipeLine(BasePipeLine):

    def __init__(self, config):
        super().__init__(config)

    def before_train_step(self, x, y):
        x, y = super().before_train_step(x, y)
        return self.model.preprocess(x, y, self.dataset.data_type)

    def before_eval_step(self, x, y):
        return self.model.preprocess(x, y, self.dataset.data_type)

    def unpreprocess_after_sample(self, samples):
        return self.model.unpreprocess(samples)
