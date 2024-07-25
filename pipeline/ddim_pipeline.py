import logging
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import wandb

from models import *
from models.ddpm import DDPM
from utils import monitor

from .ddpm_pipeline import DDPMPipeLine

__all__ = ["DDIMPipeLine"]


class DDIMPipeLine(DDPMPipeLine):

    def __init__(self, config):
        super().__init__(config)
