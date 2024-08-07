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
        self.denoise = config.sampling.denoise
        self.predict_type = config.model.predict_type
        self.support_sampling_method = ("ode", "numerical", "pc", "dpm_solver")

    @abstractmethod
    def sde(self, x, t):
        pass

    @abstractmethod
    def prior_logp(self, z):
        pass

    @abstractmethod
    def discretize(self, x, t):
        pass

    def reverse(self, probability_flow=False):
        fsde = self

        class RSDE:

            def __init__(self):
                self.fsde = fsde
                self.ode = probability_flow

            def sde(self, x, t, y, use_ema, uncond_y, guidance_scale):
                drift, diffusion = self.fsde.sde(x, t)
                score, _ = self.fsde.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
                diffusion_coef = 0.5 if self.ode else 1.0
                # logging.info(f"diffustion:{diffusion[0]}")
                drift = drift - batch_scalar_prod(diffusion_coef * diffusion**2, score)
                diffusion = 0 if self.ode else diffusion
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

    def rsde(self, x, t, y, use_ema, uncond_y, guidance_scale):
        rsde = self.reverse()
        drift, diffusion = rsde.sde(x, t, y, use_ema, uncond_y, guidance_scale)
        return drift, diffusion

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = self.ns.generate_rand_t(b)
        noise = torch.randn_like(x)
        mean, std = self.ns.marginal_coef(t)
        perturbed_x = batch_scalar_prod(mean, x) + batch_scalar_prod(std, noise)
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
            _, std = self.ns.marginal_coef(t)
            # logging.info(f"t:{t[0]}, std:{std[0]}")

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

    def cal_expected_norm(self, sigma):
        expected_norm = np.sqrt(self.img_size[0] * self.img_size[1] * self.img_channels)
        if self.predict_type != "noise":
            expected_norm /= sigma
        return expected_norm


@model_factory.register_model(name="vesde")
class VESDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)

    def sde(self, x, t):
        _, recursive_sigma_t = self.ns.recursive_cond_coef(t)
        diffusion = recursive_sigma_t
        drift = torch.zeros_like(x)
        return drift, diffusion

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        _, recursive_sigma_t = self.ns.recursive_cond_coef(t)
        f = torch.zeros_like(x)
        G = recursive_sigma_t
        return f, G


@model_factory.register_model(name="vpsde")
class VPSDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)

    def cal_expected_norm(self, sigma):
        return super().cal_expected_norm(1.0)

    def sde(self, x, t):
        _, sqrt_beta_t = self.ns.recursive_cond_coef(t)
        drift = -0.5 * batch_scalar_prod(sqrt_beta_t**2, x)
        diffusion = sqrt_beta_t
        return drift, diffusion

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def discretize(self, x, t):
        sqrt_alpha_t, sqrt_beta_t = self.ns.recursive_cond_coef(t)
        f = batch_scalar_prod(sqrt_alpha_t, x) - x
        G = sqrt_beta_t
        return f, G
