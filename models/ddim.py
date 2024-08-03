import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import model_factory
from .ddpm import DDPM


@model_factory.register_model(name="ddim")
class DDIM(DDPM):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.eta = config.test.eta
        self.sampling_steps = config.sampling.sampling_steps

    @torch.no_grad()
    def sample(self, batch_size, y=None, use_ema=True, uncond_y=None, guidance_scale=0.0):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        T = len(self.betas)
        interval = T // self.sampling_steps
        step_list = np.asarray(list(range(0, T, interval)), dtype=np.int64)
        # logging.info(step_list)
        step_list_prev = np.concatenate([[-1], step_list[:-1]])
        # logging.info(step_list_prev)

        x = self.prior_sampling(batch_size, self.device)
        for t in reversed(range(0, self.sampling_steps)):
            x = self.ddim_enoising_step2(x, step_list[t], step_list_prev[t], y, use_ema)
        return x.detach()

    @torch.no_grad()
    def ddim_enoising_step2(self, x, t, prev_t, y, use_ema, uncond_y, guidance_scale):
        t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
        alphas_cumprod_t = self.alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        alphas_cumprod_prev_t = self.alphas_cumprod[prev_t] if prev_t > 0 else self.alphas_cumprod[0]
        sqrt_one_minus_alphas_cumprod_prev_t = 1 - alphas_cumprod_prev_t
        sigma_tau = (
            self.eta
            * sqrt_one_minus_alphas_cumprod_prev_t
            / sqrt_one_minus_alphas_cumprod_t
            * torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_prev_t)
        )
        noise, _ = self.sampling_predict(x, t_batch, y, use_ema, uncond_y, guidance_scale)

        # pred x_0
        pred_x_0 = (x - sqrt_one_minus_alphas_cumprod_t * noise) / torch.sqrt(alphas_cumprod_t)
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)
        pred_sample_direction = torch.sqrt(1 - alphas_cumprod_prev_t - sigma_tau**2) * noise
        x_minus_one = pred_x_0 * torch.sqrt(alphas_cumprod_prev_t) + pred_sample_direction
        if t > 0:
            x_minus_one = x_minus_one + sigma_tau * torch.randn_like(x)
        return x_minus_one

    # @torch.no_grad()
    # def ddim_enoising_step(self, x, scalar_t, scalar_prev_t, y, use_ema):
    #     t_batch = torch.tensor([scalar_t], device=x.device).repeat(x.shape[0])
    #     t_batch_prev = torch.tensor([scalar_prev_t], device=x.device).repeat(x.shape[0])
    #     alphas_cumprod_t = self.extract(self.alphas_cumprod, t_batch, x.shape)
    #     alphas_cumprod_prev_t = self.extract(self.alphas_cumprod, t_batch_prev, x.shape)
    #     sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
    #     sqrt_one_minus_alphas_cumprod_prev_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t_batch_prev, x.shape)
    #     sigma_tau = (
    #         self.eta
    #         * sqrt_one_minus_alphas_cumprod_prev_t
    #         / sqrt_one_minus_alphas_cumprod_t
    #         * torch.sqrt(1 - alphas_cumprod_t / alphas_cumprod_prev_t)
    #     )

    #     noise, _ = self.predict(x, t_batch, y, use_ema)

    #     # pred x_0
    #     pred_x_0 = (x - sqrt_one_minus_alphas_cumprod_t * noise) / torch.sqrt(alphas_cumprod_t)
    #     pred_x_0 = torch.clamp(pred_x_0, -1, 1)

    #     x_minus_one = pred_x_0 * torch.sqrt(alphas_cumprod_prev_t) + torch.sqrt(1 - alphas_cumprod_prev_t - sigma_tau**2) * noise
    #     if scalar_t > 0:
    #         x_minus_one = x_minus_one + sigma_tau * torch.randn_like(x)
    #     return x_minus_one
