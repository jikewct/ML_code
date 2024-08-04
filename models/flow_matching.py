import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy import integrate

from lib.tensor_trans import *
from models.noise_schedule import BaseNoiseSchedule
from models.sample import *
from network import *

from . import model_factory, model_utils
from .base_model import BaseModel
from .ema import EMA
from .noise_schedule import *


@model_factory.register_model(name="flowMatching")
class FlowMatching(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.continuous = True
        self.support_sampling_method = ("ode", "rk45")

    # def init_schedule(self, config) -> RectifiedNoiseSchedule:
    #     return RectifiedNoiseSchedule(self.continuous, self.device)

    ### t: 0 ---> T
    ### x: x ---> noise
    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = torch.rand((b,), device=x.device)
        # logging.info(x.shape, t.shape)
        x_t, x_0 = self.generate_trajectory_point(x, t)
        speed_vf = x_0 - x
        predict_vf, extra_info = self.predict(x_t, t, y)
        extra_info.update({"t": t, "noise": x_0})
        return predict_vf, speed_vf, extra_info

    def generate_trajectory_point(self, x, t):
        x_0 = self.prior_sampling(x.shape[0], x.device)
        t = t.view(x.shape[0], *([1] * len(x.shape[1:])))
        # logging.info(x.shape, x_0.shape, t.shape)
        x_t = (1 - t) * x + t * x_0
        return x_t, x_0

    def rode(self, x, t, y, use_ema, uncond_y, guidance_scale):
        # t = self.ns.T - t
        predict_vf, _ = self.sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        # return -predict_vf
        return -predict_vf
