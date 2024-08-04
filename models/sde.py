import logging
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from scipy import integrate

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import DPM_Solver, NoiseScheduleVP, interpolate_fn

from . import model_factory, model_utils
from .base_model import BaseModel
from .ema import EMA
from .noise_schedule import *

"""
    0   --------> T
    x   --------> noise
snr_min --------> snr_max
"""


class SDE(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.model_config = config.model[config.model.name]
        # self.num_scales = self.model_config.num_scales
        self.denoise = config.sampling.denoise
        self.predict_type = config.model.predict_type
        self.support_sampling_method = ("ode", "rk45", "pc", "dpm_solver")

    @abstractmethod
    def sde(self, x, t):
        pass

    @abstractmethod
    def prior_sampling(self, batch_size, device):
        pass

    @abstractmethod
    def prior_logp(self, z):
        pass

    def marginal_std(self, t):
        return self.ns.marginal_coef(t)[1]

    def marginal_prob(self, x, t):

        mean_coeff, std = self.ns.marginal_coef(t)
        mean = batch_scalar_prod(mean_coeff, x)
        return mean, std

    def get_alpha(self, t):
        return self.ns.get_alpha(t)

    def get_beta(self, t):
        return self.ns.get_beta(t)

    def discretize(self, x, t):
        dt = self.ns.T / self.ns.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.divice))
        return f, G

    def reverse(self, probability_flow=False):
        fsde = self

        class RSDE:

            def __init__(self):
                self.fsde = fsde
                self.ode = probability_flow

            def sde(self, x, t, y, use_ema, uncond_y, guidance_scale):
                sigma = self.fsde.marginal_prob(x, t)[1][0]
                drift, diffusion = self.fsde.sde(x, t)
                score, extra_info = self.fsde.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
                diffusion_coef = 0.5 if self.ode else 1.0
                drift = drift - batch_scalar_prod(diffusion_coef * diffusion**2, score)
                diffusion = 0 if self.ode else diffusion
                ###debug sampling
                # expected_norm = self.fsde.cal_pred_and_expected_norm(sigma)
                # extra_info.update({"log": {"sigma": sigma}})
                # self.fsde.debug_sampling(x, score, extra_info, expected_norm)

                return drift, diffusion

            def discretize(self, x, t, y, use_ema, uncond_y, guidance_scale):
                f, G = self.fsde.discretize(x, t)
                score, _ = self.fsde.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
                G_coef = 0.5 if self.ode else 1.0
                rev_f = f - batch_scalar_prod(G_coef * G**2, score)
                rev_G = torch.zeros_like(G) if self.ode else G
                return rev_f, rev_G

        return RSDE()

    def rode(self, x, t, y, use_ema, uncond_y, guidance_scale):
        rsde = self.reverse(probability_flow=True)
        drift, _ = rsde.sde(x, t, y, use_ema, uncond_y, guidance_scale)
        return drift

    def generate_t(self, batch_size, device):

        t = torch.rand((batch_size,), device=device) * (self.ns.T - self.ns.EPS) + self.ns.EPS
        # if self.continuous: return t
        # t = self.timesteps[self.convert_t_cnt2dct(t)]
        return t

    def cal_target_score(self, x, x_t, t):
        pass

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = self.generate_t(b, x.device)
        # logging.info(x.shape, t.shape)

        noise = torch.randn_like(x)
        mean, std = self.marginal_prob(x, t)
        perturbed_x = mean + batch_scalar_prod(std, noise)
        preds, extra_info = self.predict(perturbed_x, t, y)
        if self.predict_type == "noise":
            target = noise
        elif self.predict_type == "score":
            target = batch_scalar_prod(1 / std, -noise)
        extra_info.update({"t": t, "noise": noise})
        return preds, target, extra_info

    def score_sampling_predict(self, x, t, y, use_ema, uncond_y, guidance_scale):
        preds, extra_info = super().sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        if self.predict_type == "noise":
            std = self.marginal_std(t)
            # logging.info(f"std:{std[0]}")
            preds = batch_scalar_prod(1 / std, -preds)
        return preds, extra_info

    def before_predict(self, x, t, y, use_ema):
        if self.continuous:
            cvt_t = t * self.ns.N
        else:
            cvt_t = self.convert_t_cnt2dct(t)
        # logging.info(f"t_shape:{cvt_t.shape}, t_max:{cvt_t.max()}, t_min:{cvt_t.min()}")
        return super().before_predict(x, cvt_t, y, use_ema)

    #### (0, 1.0)  ---> [0, N-1]
    def convert_t_cnt2dct(self, t):
        return (t * (self.ns.N - 1) / self.ns.T).round().long()

    # @torch.no_grad()
    # def sample(self, batch_size, y=None, use_ema=True, uncond_y=None, guidance_scale=0.0):

    #     z = self.prior_sampling(batch_size, self.device)
    #     x = self.sampler.sample(z, y, use_ema, uncond_y, guidance_scale)
    #     return x


@model_factory.register_model(name="vesde")
class VESDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.predict_type = config.model.predict_type
        # self.N = config.model.num_scales
        self.sigma_min = config.model.sigma_min
        self.sigma_max = config.model.sigma_max
        sigma_list = torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.ns.N)
        self.discrete_sigmas = torch.exp(sigma_list).to(config.device)
        self.discrete_sigmas_prev = torch.concat((torch.tensor(0, device=config.device).view(-1), self.discrete_sigmas[1:])).to(config.device)

    def get_alpha(self, t):
        return torch.ones_like(t)

    def sde(self, x, t):
        sigma_t = self.marginal_prob(x, t)[1]
        diffusion = sigma_t * torch.sqrt(torch.tensor(2 * np.log(self.sigma_max / self.sigma_min), device=x.device))
        drift = torch.zeros_like(x)
        return drift, diffusion

    def marginal_std(self, t):
        return torch.pow(self.sigma_max / self.sigma_min, t) * self.sigma_min

    def marginal_prob(self, x, t):
        sigma_t = self.marginal_std(t)
        return x, sigma_t

    def prior_sampling(self, batch_size, device):
        z = torch.randn(batch_size, self.img_channels, *self.img_size, device=device) * self.sigma_max
        return z

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step].to(x.device)
        sigma_prev = self.discrete_sigmas_prev[t_step].to(x.device)
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - sigma_prev**2)
        return f, G


@model_factory.register_model(name="vpsde")
class VPSDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)

    # def init_schedule(self, config):
    #     class_name = VPSDE.__name__.lower()
    #     model_config = config.model[class_name]
    #     if model_config.schedule_type == "sd":
    #         noise_schedule = VPSDNoiseSchedule(self.continuous, self.device, **model_config)
    #     elif model_config.schedule_type == "linear":
    #         noise_schedule = VPLinearNoiseSchedule(self.continuous, self.device, **model_config)
    #     else:
    #         raise NotImplementedError(f"{model_config.schedule_type} schedule not implemented")
    #     return noise_schedule

    def cal_expected_norm(self, sigma):
        return super().cal_expected_norm(1.0)

    def sde(self, x, t):
        beta_t = self.get_beta(t)
        drift = -0.5 * batch_scalar_prod(beta_t, x)
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def prior_sampling(self, batch_size, device):
        z = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        return z

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def discretize(self, x, t):
        alpha_t, beta_t = self.get_alpha(t), self.get_beta(t)
        f = batch_scalar_prod(torch.sqrt(alpha_t), x) - x
        G = torch.sqrt(beta_t)
        return f, G
