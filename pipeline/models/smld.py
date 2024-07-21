import logging
import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from . import model_utils
from .base_model import BaseModel
from .ema import EMA
from .model_utils import *
from .network import ncsnv2, net_utils


@model_utils.register_model(name="smld")
class SMLD(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.anneal_power = config.training.anneal_power
        self.n_steps_each = config.sampling.n_steps_each
        self.step_lr = config.sampling.step_lr
        self.denoise = config.sampling.denoise
        self.final_only = config.sampling.final_only
        self.sigma_max = config.model.sigma_max
        self.sigma_min = config.model.sigma_min
        self.predict_type = config.model.predict_type
        self.union_tau = config.model.union_tau
        self.union_threshold = config.model.union_threshold

    def init_coefficient(self, config):
        self.sigmas = net_utils.get_sigmas(config)

    @property
    def T(self):
        return len(self.sigmas)

    def marginal_std(self, t):
        return self.sigmas[t]

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = torch.randint(0, len(self.sigmas), (b, ), device=x.device)
        #used_sigmas = self.extract(self.sigmas, t, x.shape)
        used_sigmas = self.sigmas[t].view(b, *([1] * len(x.shape[1:])))
        noise = torch.randn_like(x)
        perturbed_x = x + noise * used_sigmas
        scores, extra_info = self.predict(perturbed_x, t)
        scores = scores * used_sigmas
        #logging.info(scores.shape, noise.shape)
        extra_info.update({'t': t, 'noise': noise})
        return scores, -noise, extra_info

    def SNR(self, t):
        return 1 / (self.sigmas[t[0]]**2).item()

    def predict(self, x, t, y=None, use_ema=False):
        preds, extra_info = super().predict(x, t, y, use_ema)
        if self.predict_type == "noise":
            used_sigmas = self.sigmas[t].view(x.shape[0], *([1] * len(x.shape[1:])))
            preds = preds / used_sigmas
        # elif self.predict_type == "union":
        #     used_sigmas = self.sigmas[t].view(x.shape[0], *([1] * len(x.shape[1:])))
        #     # sigmid( tau *(x - threshold))
        #     sigmoid_t = torch.sigmoid(self.union_tau * (t - self.union_threshold))[:, None, None, None]
        #     preds = -(((x - preds) / used_sigmas) * sigmoid_t + (preds * (1 - sigmoid_t))) / used_sigmas

        if self.mode == ModeEnum.SAMPLING:
            self.debug_sampling(x, t, preds, extra_info)
        return preds, extra_info

    def prior_sampling(self, batch_size, device):
        return torch.randn(batch_size, self.img_channels, *self.img_size, device=device) * self.sigma_max

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, steps=1000):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        #logging.info(batch_size, self.img_channels, self.img_size)
        x = self.prior_sampling(batch_size, device)
        for c, sigma in tqdm.tqdm(enumerate(self.sigmas), desc="sampling level", total=len(self.sigmas), leave=False):
            t = (torch.ones(batch_size, device=x.device) * c).to(torch.long)
            step_size = self.step_lr * (sigma / self.sigmas[-1])**2
            for s in tqdm.tqdm(range(self.n_steps_each), desc="sampling step", total=self.n_steps_each, leave=False):

                grad, extra_info = self.predict(x, t, y, use_ema)
                noise = torch.randn_like(x)
                x = x + step_size * grad + noise * ((2 * step_size)**(0.5))

        if self.denoise:
            t = (torch.ones(batch_size, device=x.device) * (len(self.sigmas) - 1)).to(torch.long)
            grad, extra_info = self.predict(x, t, y, use_ema)
            x = x + self.sigmas[-1]**2 * grad
        return x.detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True, steps=1000):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.detach()]

        for t in range(steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.denoising_mean(x, t_batch, y, use_ema)

            if t > 0:
                x += self.extract(self.posterior_variance, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.detach())

        return diffusion_sequence
