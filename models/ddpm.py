import logging
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from network import *

from . import model_factory, model_utils
from .base_model import BaseModel


@model_factory.register_model(name="ddpm")
class DDPM(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def parameters(self):
        return self.network.parameters()

    def init_coefficient(self, config):
        super().init_coefficient(config)
        to_torch = partial(torch.tensor, dtype=torch.float32, device=config.device)
        betas = self.generate_betas_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        # beta_t
        self.betas = to_torch(betas)
        # alpha_t
        self.alphas = to_torch(alphas)
        # alpha_t_bar
        self.alphas_cumprod = to_torch(alphas_cumprod)
        # sqrt(1/alpha_t_bar)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1.0 / alphas_cumprod))
        # sqrt(alpha_t_bar)
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        # sqrt(1/alpha_t_bar -1)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        # posterior_mean_coef1
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef1 = to_torch(posterior_mean_coef1)
        # posterior_mean_coef2
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = to_torch(posterior_mean_coef2)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1 - alphas_cumprod))
        self.reciprocal_sqrt_alphas = to_torch(np.sqrt(1 / alphas))
        self.remove_noise_coeff = to_torch(betas / np.sqrt(1 - alphas_cumprod))
        self.posterior_variance = to_torch(np.sqrt(np.append(posterior_variance[1], betas[1:])))
        self.sigma = to_torch(np.sqrt(betas))

    def generate_betas_schedule(self, config):

        num_scales = config.model.num_scales
        beta_min = config.model.beta_min
        beta_max = config.model.beta_max
        if config.model.schedule == "cosine":
            betas = self.generate_cosine_schedule(num_scales)
        else:
            betas = self.generate_linear_schedule(
                num_scales,
                beta_min * 1000 / num_scales,
                beta_max * 1000 / num_scales,
            )
        return betas

    def generate_cosine_schedule(self, T, s=0.008):

        def f(t, T):
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

        alphas = []
        f0 = f(0, T)

        for t in range(T + 1):
            alphas.append(f(t, T) / f0)
        betas = []

        for t in range(1, T + 1):
            betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

        return np.array(betas)

    def generate_linear_schedule(self, T, low, high):
        return np.linspace(low, high, T)

    @property
    def T(self):
        return len(self.betas)

    def marginal_std(self, t):
        return self.sqrt_one_minus_alphas_cumprod[t]

    def perturb_x(self, x, t, noise):
        perturb_x = self.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
        perturb_x += self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise

        return perturb_x

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        device = x.device
        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, len(self.betas), (b,), device=device)
        noise = torch.randn_like(x)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise, extra_info = self.predict(perturbed_x, t, y)
        extra_info.update({"t": t, "noise": noise})
        return estimated_noise, noise, extra_info

    def prior_sampling(self, batch_size, device):
        return torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

    def SNR(self, t):
        return (self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])).item()

    # def predict(self, x, t, y=None, use_ema=False):
    #     preds, extra_info = super().predict(x, t, y, use_ema)
    #     if self.mode == model_utils.ModeEnum.SAMPLING:
    #         self.debug_sampling(x, t, preds, extra_info)
    #     return preds, extra_info

    @torch.no_grad()
    def denoising_step(self, x, scalar_t, y, use_ema, uncond_y, guidance_scale):
        t_batch = torch.tensor([scalar_t], device=x.device).repeat(x.shape[0])
        # predict eps
        model_output, extra_info = self.sampling_predict(x, t_batch, y, use_ema, uncond_y, guidance_scale)

        # predict clipped x_0
        pred_x_0 = (
            self.extract(self.sqrt_recip_alphas_cumprod, t_batch, x.shape) * x
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t_batch, x.shape) * model_output
        )
        pred_x_0 = torch.clamp(pred_x_0, -1, 1)

        x_minus_one = (
            self.extract(self.posterior_mean_coef1, t_batch, x.shape) * pred_x_0 + self.extract(self.posterior_mean_coef2, t_batch, x.shape) * x
        )
        if scalar_t > 0:
            x_minus_one += self.extract(self.posterior_variance, t_batch, x.shape) * torch.randn_like(x)

        return x_minus_one

    def cal_expected_norm(self, sigma):
        return super().cal_expected_norm(1.0)

    @torch.no_grad()
    def sample(self, batch_size, y=None, use_ema=True, uncond_y=None, guidance_scale=0.0):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        # logging.info(batch_size, self.img_channels, self.img_size)
        x = self.prior_sampling(batch_size, self.device)

        for t in tqdm.tqdm(range(len(self.betas) - 1, -1, -1), desc="sampling", total=len(self.betas), leave=False):
            x = self.denoising_step(x, t, y, use_ema, uncond_y, guidance_scale)

        return x.detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=self.device)
        diffusion_sequence = [x.detach()]

        for t in range(len(self.betas) - 1, -1, -1):
            t_batch = torch.tensor([t], device=self.device).repeat(batch_size)
            x = self.denoising_mean(x, t_batch, y, use_ema)

            if t > 0:
                x += self.extract(self.posterior_variance, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.detach())

        return diffusion_sequence
