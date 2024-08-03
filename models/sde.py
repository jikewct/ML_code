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
        self.num_scales = self.model_config.num_scales
        self.denoise = config.sampling.denoise
        self.continuous = config.training.continuous
        self.predict_type = config.model.predict_type

    def init_coefficient(self, config):
        self.timesteps = torch.linspace(self.T, self.EPS, self.N, device=config.device)

    @property
    def T(self):
        return 1.0

    @property
    def EPS(self):
        return 1e-5

    @property
    def N(self):
        return self.num_scales

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

        class RSDE:

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

    def generate_t(self, batch_size, device):

        t = torch.rand((batch_size,), device=device) * (self.T - self.EPS) + self.EPS
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
            cvt_t = t * self.N
        else:
            cvt_t = self.convert_t_cnt2dct(t)
        # logging.info(f"t_shape:{cvt_t.shape}, t_max:{cvt_t.max()}, t_min:{cvt_t.min()}")
        return super().before_predict(x, cvt_t, y, use_ema)

    #### (0, 1.0)  ---> [0, N-1]
    def convert_t_cnt2dct(self, t):
        return (t * (self.N - 1) / self.T).round().long()

    def convert_t_dct2cnt(self, t):
        return self.timesteps[self.N - 1 - t]

    def corrector_sample(self, x, t, y, use_ema, uncond_y, guidance_scale, corrector, n_step_each, snr):
        if corrector.lower() == "ald":
            x, x_mean = self.ald_correct(x, t, y, use_ema, uncond_y, guidance_scale, n_step_each, snr)
        elif corrector.lower() == "langevin":
            x, x_mean = self.langevin_correct(x, t, y, use_ema, uncond_y, guidance_scale, n_step_each, snr)
        else:
            return x, x
        return x, x_mean

    def langevin_correct(self, x, t, y, use_ema, uncond_y, guidance_scale, n_step_each, snr):
        for i in range(n_step_each):
            grad, _ = self.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=1).mean()
            step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * self.get_alpha(t)
            x_mean = x + batch_scalar_prod(step_size, grad)
            x = x_mean + batch_scalar_prod(torch.sqrt(2 * step_size), noise)
        return x, x_mean

    def ald_correct(self, x, t, y, use_ema, uncond_y, guidance_scale, n_step_each, snr):
        for i in range(n_step_each):
            grad, _ = self.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
            noise = torch.randn_like(x)
            std = self.marginal_prob(x, t)[1]
            step_size = (snr * std) ** 2 * 2 * self.get_alpha(t)
            x_mean = x + batch_scalar_prod(step_size, grad)
            x = x_mean + batch_scalar_prod(torch.sqrt(2 * step_size), noise)
        return x, x_mean

    def predictor_sample(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale, predictor):
        if predictor.lower() == "euler":
            x, x_mean = self.eulerMaruyama_predict(rsde, x, t, y, use_ema, uncond_y, guidance_scale)
        elif predictor.lower() == "reversediffusion":
            x, x_mean = self.reverse_diffusion_predict(rsde, x, t, y, use_ema, uncond_y, guidance_scale)
        elif predictor.lower() == "ancestralsampling":
            x, x_mean = self.ancestral_sampling_predict(rsde, x, t, y, use_ema, uncond_y, guidance_scale)
        else:
            return x, x
        return x, x_mean

    def reverse_diffusion_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        f, G = rsde.discretize(x, t, y, use_ema, uncond_y, guidance_scale)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + batch_scalar_prod(G, z)
        return x, x_mean

    @abstractmethod
    def ancestral_sampling_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        pass

    def eulerMaruyama_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        dt = -self.T / rsde.N * self.N
        z = torch.randn_like(x)
        drift, diffusion = rsde.sde(x, t, y, use_ema, uncond_y, guidance_scale)
        x_mean = x + drift * dt
        x = x_mean + batch_scalar_prod(diffusion, np.sqrt(-dt) * z)
        return x, x_mean

    def pc_sample(self, batch_size, device, y, use_ema, uncond_y, guidance_scale, predictor="", corrector="", n_step_each=1, snr=0.16):
        shape = (batch_size, self.img_channels, *self.img_size)
        x = self.prior_sampling(batch_size, device)
        rsde = self.reverse()

        for i in tqdm.tqdm(range(self.N), desc="pc sample", total=self.N, leave=False):
            t = self.timesteps[i]
            vec_t = torch.ones(batch_size, device=device) * t
            x, x_mean = self.corrector_sample(x, vec_t, y, use_ema, uncond_y, guidance_scale, corrector, n_step_each, snr)
            x, x_mean = self.predictor_sample(rsde, x, vec_t, y, use_ema, uncond_y, guidance_scale, predictor)
        # x_max = x.max()
        # x_min = x.min()
        # x = (x - x_min) / (x_max - x_min + 1e-7) * 2 - 1
        return x_mean if self.denoise else x

    def ode_sample(self, batch_size, device, y, use_ema, uncond_y, guidance_scale, sampling_steps=10):
        #  shape = (batch_size, self.img_channels, *self.img_size)
        x = self.prior_sampling(batch_size, device)
        rsde = self.reverse(probability_flow=True)
        dt = -self.T / sampling_steps * self.N
        steps = np.linspace(0, self.N - 1, sampling_steps).round().astype(np.int32)
        for i in tqdm.tqdm(steps, desc="ode sample", total=sampling_steps, leave=False):
            t = self.timesteps[i]
            vec_t = torch.ones(batch_size, device=device) * t
            drift, diffusion = rsde.sde(x, vec_t, y, use_ema, uncond_y, guidance_scale)
            x = x + drift * dt
            # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x

    def rk45_sample(self, batch_size, device, y, use_ema, uncond_y, guidance_scale, **kargs):
        x = self.prior_sampling(batch_size, device)
        shape = x.shape
        rsde = self.reverse(probability_flow=True)
        solution = integrate.solve_ivp(
            self.ode_func,
            (self.N, 0),
            self.to_flattened_numpy(x),
            method="RK45",
            args=(y, rsde, shape, device, use_ema, uncond_y, guidance_scale),
            **kargs,
        )
        nfe = solution.nfev
        logging.info(f"sample nfe:{nfe}")
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        return x.detach()

    def ode_func(self, t, x, y, rsde, shape, device, use_ema, uncond_y, guidance_scale):
        x = self.from_flattened_numpy(x, shape).to(device).type(torch.float32)
        t = torch.ones(shape[0], device=device) * t / self.N
        drift, _ = rsde.sde(x, t, y, use_ema, uncond_y, guidance_scale)
        return self.to_flattened_numpy(drift)

    def dpm_solver_sample(self, batch_size, device, y, use_ema, uncond_y, guidance_scale, sampling_steps=50):
        shape = (batch_size, self.img_channels, *self.img_size)
        x = self.prior_sampling(batch_size, device)
        # x = torch.from_numpy(np.load("/home/jikewct/Dataset/coco2017/coco_256_feature/noise_1x4x32x32.npy")).to(device).float()
        # x = einops.repeat(x, "1 C H W -> B C H W", B=batch_size)
        noise_schedule = self.get_noise_schedule()
        # logging.info(batch_size, y.shape, use_ema, uncond_y.shape, guidance_scale, sampling_steps)

        def model_fn(x, t_continuous):
            # t = t_continuous * self.N
            # logging.info(t[0])
            preds, _ = self.sampling_predict(x, t_continuous, y, use_ema, uncond_y, guidance_scale)
            return preds

        dmp_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        samples = dmp_solver.sample(x, steps=sampling_steps, eps=self.EPS, T=self.T)
        return samples

    def get_noise_schedule(self):
        pass

    @torch.no_grad()
    def sample(self, batch_size, y=None, use_ema=True, uncond_y=None, guidance_scale=0.0):
        if self.sampling_method.lower() == "ode":
            x = self.ode_sample(batch_size, self.device, y, use_ema, uncond_y, guidance_scale, **self.sampling_config)
        elif self.sampling_method.lower() == "pc":
            x = self.pc_sample(batch_size, self.device, y, use_ema, uncond_y, guidance_scale, **self.sampling_config)
        elif self.sampling_method.lower() == "dpm_solver":
            x = self.dpm_solver_sample(batch_size, self.device, y, use_ema, uncond_y, guidance_scale, **self.sampling_config)
        elif self.sampling_method.lower() == "rk45":
            x = self.rk45_sample(batch_size, self.device, y, use_ema, uncond_y, guidance_scale, **self.sampling_config)

        else:
            raise NotImplementedError(f"{self.sampling_method} not yet supported.")
        return x


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
        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step].to(x.device)
        sigma_prev = self.discrete_sigmas_prev[t_step].to(x.device)
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - sigma_prev**2)
        return f, G

    def ancestral_sampling_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        t_step = self.convert_t_cnt2dct(t)
        sigma = self.discrete_sigmas[t_step]
        sigma_prev = self.discrete_sigmas_prev[t_step]
        score, _ = self.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        x_mean = x + batch_scalar_prod(sigma**2 - sigma_prev**2, score)
        std = torch.sqrt((sigma_prev**2 * (sigma**2 - sigma_prev**2)) / (sigma**2))
        noise = torch.randn_like(x)
        x = x_mean + batch_scalar_prod(std, noise)
        return x, x_mean

    # def after_predict(self, x, t, y, user_ema, preds, extra_info):
    #     std = self.marginal_std(t)
    #     if self.predict_type == "noise":
    #         preds = preds / std[:, None, None, None]
    #     return super().after_predict(x, t, y, user_ema, preds, extra_info)


@model_factory.register_model(name="vpsde")
class VPSDE(SDE):

    def __init__(self, config):
        super().__init__(config)

    def init_parameters(self, config):
        super().init_parameters(config)
        # self.N = config.model.num_scales
        self.beta_min = self.model_config.beta_min
        self.beta_max = self.model_config.beta_max
        if not self.continuous:
            self.init_discrete_parameter(config)

    def init_discrete_parameter(self, config):
        self.betas = self.generate_betas_schedule().to(config.device)
        # self.alphas = 1.0 - self.betas
        # self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=config.device), self.alphas_cumprod[:-1]])
        # # print(self.alphas_cumprod.shape, self.alphas_cumprod.dtype, self.alphas_cumprod)
        # self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        # self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_alphas_discrete = 0.5 * torch.log(1 - self.betas).cumsum(dim=0).reshape((1, -1))
        self.t_discrete = torch.linspace(self.EPS, 1.0, self.N).reshape((1, -1))

    def generate_betas_schedule(self):

        if self.model_config.schedule == "cosine":
            betas = self.generate_cosine_schedule(self.num_scales)
        if self.model_config.schedule == "sd":
            betas = self.gengerate_sd_schedule(self.num_scales, self.beta_min, self.beta_max)
        else:
            betas = self.generate_linear_schedule(self.num_scales, self.beta_min, self.beta_max)
        return betas

    def get_noise_schedule(self):
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
        return noise_schedule

    def gengerate_sd_schedule(self, num_scales, low, high):
        return torch.linspace(low**0.5, high**0.5, num_scales) ** 2

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

    def get_alpha(self, t):
        return 1.0 - self.get_beta(t)

    def get_beta(self, t):
        if self.model_config.schedule == "linear":
            beta_t = (self.beta_max - self.beta_min) * t + self.beta_min
        elif self.model_config.schedule == "sd":
            beta_t = ((self.beta_max**0.5 - self.beta_min**0.5) * t + self.beta_min**0.5) ** 2
        else:
            raise NotImplementedError
        return beta_t

    def get_alpha_cum(self, t):
        if self.continuous:
            if self.model_config.schedule == "linear":
                mean_coeff = torch.exp(-0.5 * (2 * t**2 * (self.beta_max - self.beta_min) + t * self.beta_min))
            elif self.model_config.schedule == "sd":
                a = self.beta_max**0.5 - self.beta_min**0.5
                b = self.beta_min**0.5
                mean_coeff = torch.exp(-0.5 * (1.0 / 3 * t**3 * a**2 + a * b * t ** +(b**2) * t))
            else:
                raise NotImplementedError
        else:
            log_mean_coeff = interpolate_fn(
                t.reshape((-1, 1)), self.t_discrete.clone().to(t.device), self.log_alphas_discrete.clone().to(t.device)
            ).reshape((-1,))
            mean_coeff = torch.exp(log_mean_coeff)
        return mean_coeff**2

    def cal_expected_norm(self, sigma):
        return super().cal_expected_norm(1.0)

    def sde(self, x, t):
        beta_t = self.get_beta(t)
        drift = -0.5 * batch_scalar_prod(beta_t, x)
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_std(self, t):
        return self.marginal_coef(t)[1]

    def marginal_coef(self, t):

        alpha_cum_t = self.get_alpha_cum(t)
        mean_coeff = torch.sqrt(alpha_cum_t)
        std = torch.sqrt(1.0 - mean_coeff**2)
        return mean_coeff, std

    def marginal_prob(self, x, t):

        mean_coeff, std = self.marginal_coef(t)
        mean = batch_scalar_prod(mean_coeff, x)
        return mean, std

    def prior_sampling(self, batch_size, device):
        z = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        return z

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def discretize(self, x, t):
        # t_step = self.convert_t_cnt2dct(t)
        # beta = self.betas[t_step]
        # alpha = self.alphas[t_step]
        # sqrt_beta = torch.sqrt(beta)
        alpha_t, beta_t = self.get_alpha(t), self.get_beta(t)
        f = batch_scalar_prod(torch.sqrt(alpha_t), x) - x
        G = torch.sqrt(beta_t)
        return f, G

    def ancestral_sampling_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        alpha_t, beta_t = self.get_alpha(t), self.get_beta(t)
        score, _ = self.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        x_mean = batch_scalar_prod(1 / torch.sqrt(alpha_t), x + batch_scalar_prod(beta_t, score))
        prev_t = t - 1.0 / self.N
        prev_t = torch.where(prev_t > 0, prev_t, self.EPS)
        std = torch.sqrt(beta_t * (1.0 - self.get_alpha_cum(prev_t)) / (1.0 - self.get_alpha_cum(t)))
        noise = torch.randn_like(x)
        x = x_mean + batch_scalar_prod(std, noise)
        return x, x_mean
