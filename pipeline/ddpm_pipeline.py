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
