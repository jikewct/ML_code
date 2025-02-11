from abc import abstractmethod

import numpy as np
import torch
import tqdm

from lib.tensor_trans import batch_scalar_prod
from models.sde import VESDE, VPSDE

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="pc")
class PCSample(SDESample):
    def __init__(self, model, sampling_method, predictor="", corrector="", n_step_each=1, snr=0.16) -> None:
        super().__init__(model, sampling_method)
        self.predictor = predictor
        self.corrector = corrector
        self.n_step_each = n_step_each
        self.snr = snr

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        rsde = self.model.reverse()
        for i in tqdm.tqdm(range(self.ns.N), desc="pc sample", total=self.ns.N, leave=False):
            # t = self.timesteps[i]
            # from T ---> EPS
            t = (self.ns.EPS - self.ns.T) * i / self.ns.N + self.ns.T
            self.ns.T - i / self.ns.N
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            x, x_mean = self.corrector_sample(x, vec_t, y, use_ema, uncond_y, guidance_scale, self.corrector, self.n_step_each, self.snr)
            x, x_mean = self.predictor_sample(rsde, x, vec_t, y, use_ema, uncond_y, guidance_scale, self.predictor)
        return x_mean if denoise else x

    def eulerMaruyama_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        dt = -self.ns.T / self.ns.N * self.ns.N
        z = torch.randn_like(x)
        drift, diffusion = rsde.sde(x, t, y, use_ema, uncond_y, guidance_scale)
        x_mean = x + drift * dt
        x = x_mean + batch_scalar_prod(diffusion, np.sqrt(-dt) * z)
        return x, x_mean

    def reverse_diffusion_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        f, G = rsde.discretize(x, t, y, use_ema, uncond_y, guidance_scale)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + batch_scalar_prod(G, z)
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

    def ancestral_sampling_predict(self, rsde, x, t, y, use_ema, uncond_y, guidance_scale):
        if isinstance(rsde.fsde, VESDE):
            return self.ve_ancestral_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        elif isinstance(rsde.fsde, VPSDE):
            return self.vp_ancestral_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        else:
            raise NotImplementedError(f"{rsde.fsde.__class__.__name__} is not implemented")

    def ve_ancestral_sampling_predict(self, x, t, y, use_ema, uncond_y, guidance_scale):
        _, sigma_t = self.ns.marginal_coef(t)
        prev_t = t - 1.0 / self.ns.N
        prev_t = torch.where(prev_t > 0, prev_t, self.ns.EPS)
        _, sigma_prev_t = self.ns.marginal_coef(prev_t)
        score, _ = self.model.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        _, recursive_sigma_t = self.ns.recursive_cond_coef(t)
        x_mean = x + batch_scalar_prod(recursive_sigma_t**2, score)
        std = sigma_prev_t * recursive_sigma_t / sigma_t
        noise = torch.randn_like(x)
        x = x_mean + batch_scalar_prod(std, noise)
        return x, x_mean

    def vp_ancestral_sampling_predict(self, x, t, y, use_ema, uncond_y, guidance_scale):
        # alpha_t, beta_t = self.ns.get_alpha(t), self.ns.get_beta(t)
        sqrt_alpha_t, sqrt_beta_t = self.ns.recursive_cond_coef(t)
        score, _ = self.model.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
        x_mean = batch_scalar_prod(1 / sqrt_alpha_t, x + batch_scalar_prod(sqrt_beta_t**2, score))
        prev_t = t - 1.0 / self.ns.N
        prev_t = torch.where(prev_t > 0, prev_t, self.ns.EPS)
        std = sqrt_beta_t * (1.0 - self.ns.marginal_coef(prev_t)[0]) / (1.0 - self.ns.marginal_coef(t)[0])
        noise = torch.randn_like(x)
        x = x_mean + batch_scalar_prod(std, noise)
        return x, x_mean

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
            grad, _ = self.model.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=1).mean()
            sqrt_alpha_t = self.ns.recursive_cond_coef(t)[0]
            step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * sqrt_alpha_t**2
            x_mean = x + batch_scalar_prod(step_size, grad)
            x = x_mean + batch_scalar_prod(torch.sqrt(2 * step_size), noise)
        return x, x_mean

    def ald_correct(self, x, t, y, use_ema, uncond_y, guidance_scale, n_step_each, snr):
        for i in range(n_step_each):
            grad, _ = self.model.score_sampling_predict(x, t, y, use_ema, uncond_y, guidance_scale)
            noise = torch.randn_like(x)
            std = self.ns.marginal_coef(t)[1]
            sqrt_alpha_t = self.ns.recursive_cond_coef(t)[0]
            step_size = (snr * std) ** 2 * 2 * sqrt_alpha_t**2
            x_mean = x + batch_scalar_prod(step_size, grad)
            x = x_mean + batch_scalar_prod(torch.sqrt(2 * step_size), noise)
        return x, x_mean
