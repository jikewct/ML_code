import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from ..utils import monitor
from . import model_utils
from .base_model import BaseModel
from .ema import EMA
from .model_utils import ModeEnum
from .network import ncsnv2, net_utils
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
        self.N = config.sampling.sample_steps
        self.denoise = config.sampling.denoise
        self.snr = config.sampling.snr
        self.n_step_each = config.sampling.n_steps_each
        self.sampling_method = config.sampling.method
        self.predictor = config.sampling.predictor
        self.corrector = config.sampling.corrector
        self.continuous = config.training.continuous

    def init_coefficient(self, config):
        self.timesteps = torch.linspace(self.T, self.EPS, self.N, device=config.device)

    @property
    def T(self):
        return 1.

    @property
    def CNT_SCALE(self):
        return 999

    @property
    def EPS(self):
        return 1e-5

    @abstractmethod
    def sde(self, x, t):
        pass

    @abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abstractmethod
    def marginal_std(self, t):
        pass

    @abstractmethod
    def prior_sampling(self, batch_size, device):
        pass

    @abstractmethod
    def prior_logp(self, z):
        pass

    @abstractmethod
    def get_alpha(self, t):
        pass

    def discretize(self, x, t):
        dt = self.T / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.divice))
        return f, G

    def reverse(self, probability_flow=False):
        fsde = self

        class RSDE():

            def __init__(self):
                self.fsde = fsde
                self.ode = probability_flow

            @property
            def T(self):
                return self.fsde.T

            @property
            def N(self):
                return self.fsde.N

            @property
            def EPS(self):
                return self.fsde.EPS

            def sde(self, x, t, y=None, use_ema=True):
                sigma = self.fsde.marginal_prob(x, t)[1][0]
                drift, diffusion = self.fsde.sde(x, t)
                score, extra_info = self.fsde.predict(x, t, y, use_ema)
                diffusion_coef = 0.5 if self.ode else 1.
                drift = drift - diffusion[:, None, None, None]**2 * score * diffusion_coef
                diffusion = 0 if self.ode else diffusion
                ###debug sampling
                expected_norm = self.fsde.cal_pred_and_expected_norm(sigma)
                extra_info.update({'log': {'sigma': sigma}})
                #self.fsde.debug_sampling(x, score, extra_info, expected_norm)

                return drift, diffusion

            def discretize(self, x, t, y=None, use_ema=True):
                f, G = self.fsde.discretize(x, t)
                score, _ = self.fsde.predict(x, t, y=None, use_ema=True)
                G_coef = 0.5 if self.ode else 1.
                rev_f = f - G[:, None, None, None]**2 * score * G_coef
                rev_G = torch.zeros_like(G) if self.ode else G
                return rev_f, rev_G

        return RSDE()

    def generate_t(self, batch_size, device):

        t = torch.rand((batch_size, ), device=device) * (self.T - self.EPS) + self.EPS
        #if self.continuous: return t
        #t = self.timesteps[self.convert_t_cnt2dct(t)]
        return t

    def cal_target_score(self, x, x_t, t):
        pass

    def forward(self, x, y=None):

        b, c, h, w = x.shape
        t = self.generate_t(b, x.device)
        #logging.info(x.shape, t.shape)

        noise = torch.randn_like(x)
        mean, std = self.marginal_prob(x, t)
        perturbed_x = mean + std[:, None, None, None] * noise
        scores, extra_info = self.predict(perturbed_x, t, y)
        scores = scores * std[:, None, None, None]
        extra_info.update({'t': t, 'noise': noise})
        return scores, -noise, extra_info

    def convert_predict_t(self, t):
        if self.continuous:
            t = t * self.CNT_SCALE
        else:
            t = self.convert_t_cnt2dct(t)
        return t

    def inverse_predict_t(self, t):
        if self.continuous:
            t = t / self.CNT_SCALE
        else:
            t = self.convert_t_dct2cnt(t)
        return t

    def predict(self, x, t, y=None, use_ema=False, debug_sampling=True):
        if self.continuous:
            cvt_t = t * self.CNT_SCALE
        else:
            cvt_t = self.convert_t_cnt2dct(t)
        preds, extra_info = super().predict(x, cvt_t, y, use_ema)
        preds = self.post_predict(x, preds, t, y)

        if self.mode == ModeEnum.SAMPLING and debug_sampling:
            self.debug_sampling(x, t, preds, extra_info)
        return preds, extra_info

    def post_predict(self, x, preds, t, y=None):
        return preds

    def convert_t_cnt2dct(self, t):
        return (t * (self.N - 1) / self.T).long()

    def convert_t_dct2cnt(self, t):
        return self.timesteps[self.N - 1 - t]

    def corrector_sample(self, x, t, y=None, use_ema=True):
        if self.corrector.lower() == "ald":
            x, x_mean = self.ald_correct(x, t, y, use_ema)
        elif self.corrector.lower() == "langevin":
            x, x_mean = self.langevin_correct(x, t, y, use_ema)
        else:
            return x, x
        return x, x_mean

    def langevin_correct(self, x, t, y=None, use_ema=True):
        for i in range(self.n_step_each):
            grad, _ = self.predict(x, t, y, use_ema)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=1).mean()
            step_size = (self.snr * noise_norm / grad_norm)**2 * 2 * self.get_alpha(t)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(2 * step_size)[:, None, None, None]
        return x, x_mean

    def ald_correct(self, x, t, y=None, use_ema=True):
        for i in range(self.n_step_each):
            grad, _ = self.predict(x, t, y, use_ema)
            noise = torch.randn_like(x)
            std = self.marginal_prob(x, t)[1]
            step_size = (self.snr * std)**2 * 2 * self.get_alpha(t)
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(2 * step_size)[:, None, None, None]
        return x, x_mean

    def predictor_sample(self, rsde, x, t, y=None, use_ema=True):
        if self.predictor.lower() == "euler":
            x, x_mean = self.eulerMaruyama_predict(rsde, x, t, y, use_ema)
        elif self.predictor.lower() == "reversediffusion":
            x, x_mean = self.reverse_diffusion_predict(rsde, x, t, y, use_ema)
        elif self.predictor.lower() == "ancestralsampling":
            x, x_mean = self.ancestral_sampling_predict(rsde, x, t, y, use_ema)
        else:
            return x, x
        return x, x_mean

    def reverse_diffusion_predict(self, rsde, x, t, y=None, use_ema=True):
        f, G = rsde.discretize(x, t, y, use_ema)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean

    @abstractmethod
    def ancestral_sampling_predict(self, rsde, x, t, y=None, use_ema=True):
        pass

    def eulerMaruyama_predict(self, rsde, x, t, y=None, use_ema=True):
        dt = -self.T / rsde.N
        z = torch.randn_like(x)
        drift, diffusion = rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean

    def pc_sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        shape = (batch_size, self.img_channels, *self.img_size)
        x = self.prior_sampling(batch_size, device)
        rsde = self.reverse()

        for i in tqdm.tqdm(range(self.N), desc="pc sample", leave=False):
            t = self.timesteps[i]
            vec_t = torch.ones(batch_size, device=device) * t
            x, x_mean = self.corrector_sample(x, vec_t, y, use_ema)
            x, x_mean = self.predictor_sample(rsde, x, vec_t, y, use_ema)
        # x_max = x.max()
        # x_min = x.min()
        # x = (x - x_min) / (x_max - x_min + 1e-7) * 2 - 1
        return x_mean if self.denoise else x

    def ode_sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        pass

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, steps=10):
        if self.sampling_method.lower() == "ode":
            x = self.ode_sample(batch_size, device, y, use_ema)
        elif self.sampling_method.lower() == "pc":
            x = self.pc_sample(batch_size, device, y, use_ema)
        else:
            raise NotImplementedError(f"{self.sampling_method} not yet supported.")
        return x


@model_utils.register_model(name="vesde")
class VESDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.predict_type = config.model.predict_type
        self.N = config.model.num_scales
        self.sigma_min = config.model.sigma_min
        self.sigma_max = config.model.sigma_max
        sigma_list = torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N)
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
        return -N / 2. * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step].to(x.device)
        sigma_prev = self.discrete_sigmas_prev[t_step].to(x.device)
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - sigma_prev**2)
        return f, G

    def ancestral_sampling_predict(self, rsde, x, t, y=None, use_ema=True):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step]
        sigma_prev = self.discrete_sigmas_prev[t_step]
        score, _ = self.predict(x, t, y, use_ema)
        x_mean = x + (sigma**2 - sigma_prev**2)[:, None, None, None] * score
        std = torch.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def post_predict(self, x, preds, t, y=None):
        std = self.marginal_std(t)
        if self.predict_type == "noise":
            preds = preds / std[:, None, None, None]
        return preds


@model_utils.register_model(name="vpsde")
class VPSDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        self.N = config.model.num_scales
        self.beta_min = config.model.beta_min
        self.beta_max = config.model.beta_max
        self.betas = self.generate_betas_schedule(config).to(config.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def generate_betas_schedule(self, config):

        num_scales = config.model.num_scales
        beta_min = config.model.beta_min
        beta_max = config.model.beta_max
        # if config.model.schedule == "cosine":
        #     betas = self.generate_cosine_schedule(num_scales)
        # else:
        betas = self.generate_linear_schedule(
            num_scales,
            beta_min * 1000 / num_scales,
            beta_max * 1000 / num_scales,
        )
        return betas

    def generate_cosine_schedule(self, T, s=0.008):

        def f(t, T):
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2))**2

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

    def get_alpha(self, t):
        return self.alphas[(t * (self.N - 1) / self.T).long()]

    def cal_expected_norm(self, sigma):
        return super().cal_pred_and_expected_norm(1.)

    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_std(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, batch_size, device):
        z = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        return z

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.

    def discretize(self, x, t):
        t_step = self.convert_t_cnt2dct(t)
        beta = self.betas[t_step]
        alpha = self.alphas[t_step]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def ancestral_sampling_predict(self, rsde, x, t, y=None, use_ema=True):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step].to(x.device)
        sigma_prev = self.discrete_sigmas_prev[t_step].to(x.device)
        score, _ = self.predict(x, t, y, use_ema)
        x_mean = x + (sigma**2 - sigma_prev**2)[:, None, None, None] * score
        std = torch.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean
