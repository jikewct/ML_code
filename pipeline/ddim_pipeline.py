import logging
from abc import ABC, abstractmethod

import imageio
import numpy as np
import torch
import wandb

from .ddpm_pipeline import DDPMPipeLine
from .models import model_utils
from .models.ddpm import DDPM
from .utils import monitor

__all__ = ["DDIMPipeLine"]


class DDIMPipeLine(DDPMPipeLine):

    def __init__(self, config):
        super().__init__(config)
