import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from scipy import integrate

from . import model_utils
from .base_model import BaseModel
from .ema import EMA
from .model_utils import *
from .network import ncsnpp, ncsnv2, net_utils, lora_ncsnpp


@model_utils.register_model(name="flowMatching")
class FlowMatching(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.sample_method = config.sampling.method
        self.num_scales = config.model.num_scales
        self.rtol = config.sampling.rtol
        self.atol = config.sampling.atol

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
        x_t = (1 - t) * x_0 + t * x
        return x_t, x_0

    def predict(self, x, t, y=None, use_ema=False):
        scaled_t = t * self.CNT_SCALE
        speed_vf, extra_info = super().predict(x, scaled_t, y, use_ema)
        if self.mode == ModeEnum.SAMPLING:
            self.debug_sampling(x, t, speed_vf, extra_info)
        return speed_vf, extra_info

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        if self.sample_method == "rk45":
            return self.rk45_sample(batch_size, device, y, use_ema)
        elif self.sample_method == "ode":
            return self.ode_sample(batch_size, device, y, use_ema, steps=steps)
        return self.ode_sample(batch_size, device, y, use_ema)

    @torch.no_grad()
    def ode_sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        time_steps = np.linspace(self.EPS, self.T, steps)
        step_size = 1 / steps
        x = self.prior_sampling(batch_size, device)
        for t_index in tqdm.tqdm(
            range(steps), desc="sampling level", total=steps, leave=False
        ):
            num_t = t_index / steps * (self.T - self.EPS) + self.EPS
            t = torch.ones((batch_size,), device=x.device) * num_t
            predict_vf, _ = self.predict(x, t, y, use_ema)
            x = x + step_size * predict_vf
        return x.detach()

    @torch.no_grad()
    def rk45_sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        x = self.prior_sampling(batch_size, device).detach().clone()
        shape = x.shape
        solution = integrate.solve_ivp(
            self.ode_func,
            (self.EPS, self.T),
            self.to_flattened_numpy(x),
            rtol=self.rtol,
            atol=self.atol,
            method="RK45",
            args=(shape, device),
        )
        nfe = solution.nfev
        logging.info(f"sample nfe:{nfe}")
        x = (
            torch.tensor(solution.y[:, -1])
            .reshape(shape)
            .to(device)
            .type(torch.float32)
        )
        return x.detach()

    def ode_func(self, t, x, shape, device):
        x = self.from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=device) * t
        speed_vf, _ = self.predict(x, vec_t, use_ema=True)
        return self.to_flattened_numpy(speed_vf)

    def prior_sampling(self, batch_size, device):
        return torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        # return torch.ones(batch_size, self.img_channels, *self.img_size, device=device)
