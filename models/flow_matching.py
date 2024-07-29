import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy import integrate

from network import lora_ncsnpp, ncsnpp, ncsnv2, uvit
from network.layers import layer_utils

from . import model_factory, model_utils
from .base_model import BaseModel
from .ema import EMA


@model_factory.register_model(name="flowMatching")
class FlowMatching(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.sample_method = config.sampling.method
        self.sampling_steps = config.sampling.sampling_steps
        self.sample_config = config.sampling[self.sample_method]
        # self.num_scales = config.model.num_scales
        # self.rtol = config.sampling.rtol
        # self.atol = config.sampling.atol

    def init_coefficient(self, config):
        pass

    def marginal_std(self, t):
        return torch.tensor([1.0])

    @property
    def T(self):
        return 1.0

    @property
    def CNT_SCALE(self):
        # return self.num_scales - 1
        return 999

    @property
    def EPS(self):
        return 1e-3

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = torch.rand((b,), device=x.device)
        # logging.info(x.shape, t.shape)
        x_t, x_0 = self.generate_trajectory_point(x, t)
        speed_vf = x - x_0
        predict_vf, extra_info = self.predict(x_t, t, y)
        extra_info.update({"t": t, "noise": x_0})
        return predict_vf, speed_vf, extra_info

    def generate_trajectory_point(self, x, t):
        x_0 = self.prior_sampling(x.shape[0], x.device)
        t = t.view(x.shape[0], *([1] * len(x.shape[1:])))
        # logging.info(x.shape, x_0.shape, t.shape)
        x_t = (1 - t) * x_0 + t * x
        return x_t, x_0

    def predict(self, x, t, y=None, use_ema=False):
        scaled_t = t * self.CNT_SCALE
        speed_vf, extra_info = super().predict(x, scaled_t, y, use_ema)
        if self.mode == model_utils.ModeEnum.SAMPLING:
            self.debug_sampling(x, t, speed_vf, extra_info)
        return speed_vf, extra_info

    @torch.no_grad()
    def sample(self, batch_size, y=None, use_ema=True):
        if self.sample_method == "rk45":
            return self.rk45_sample(batch_size, self.device, y, use_ema)
        elif self.sample_method == "ode":
            return self.ode_sample(batch_size, self.device, y, use_ema, sampling_steps=self.sampling_steps)
        return self.ode_sample(batch_size, self.device, y, use_ema)

    @torch.no_grad()
    def ode_sample(self, batch_size, device, y=None, use_ema=True, sampling_steps=10):
        time_steps = np.linspace(self.EPS, self.T, sampling_steps)
        step_size = 1 / sampling_steps
        x = self.prior_sampling(batch_size, device)
        for t_index in tqdm.tqdm(range(sampling_steps), desc="sampling level", total=sampling_steps, leave=False):
            num_t = t_index / sampling_steps * (self.T - self.EPS) + self.EPS
            t = torch.ones((batch_size,), device=x.device) * num_t
            predict_vf, _ = self.predict(x, t, y, use_ema=use_ema)
            x = x + step_size * predict_vf
        return x.detach()

    @torch.no_grad()
    def rk45_sample(self, batch_size, device, y=None, use_ema=True):
        x = self.prior_sampling(batch_size, device).detach().clone()
        shape = x.shape
        solution = integrate.solve_ivp(
            self.ode_func,
            (self.EPS, self.T),
            self.to_flattened_numpy(x),
            method="RK45",
            args=(shape, device, use_ema),
            **self.sample_config,
        )
        nfe = solution.nfev
        logging.info(f"sample nfe:{nfe}")
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        return x.detach()

    def ode_func(self, t, x, shape, device, use_ema):
        x = self.from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=device) * t
        speed_vf, _ = self.predict(x, vec_t, use_ema=use_ema)
        return self.to_flattened_numpy(speed_vf)

    def prior_sampling(self, batch_size, device):
        return torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        # return torch.ones(batch_size, self.img_channels, *self.img_size, device=device)
