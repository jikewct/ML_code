import logging
from abc import abstractmethod

import einops
import numpy as np
import torch
import torchsde
import tqdm
from scipy import integrate

from lib.tensor_trans import *

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="numerical")
class NumericalSample(SDESample):
    def __init__(self, model, sampling_method, equation_type="sde", method="euler", sampling_steps=50, **kargs) -> None:
        super().__init__(model, sampling_method)
        self.kargs = kargs
        self.method = method
        self.equation_type = equation_type
        self.sampling_steps = sampling_steps

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        if self.equation_type == "ode":
            return self._ode_sample(x, y, use_ema, uncond_y, guidance_scale)
        elif self.equation_type == "sde":
            return self._sde_sample(x, y, use_ema, uncond_y, guidance_scale)

    def _ode_sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        shape = x.shape
        t_span = (self.ns.T, self.ns.EPS)
        if not self.model.continuous:
            t_span = (self.ns.N / self.ns.T, 0)
        solution = integrate.solve_ivp(
            self._ode_func,
            t_span,
            to_flattened_numpy(x),
            method=self.method.upper(),
            args=(y, shape, x.device, use_ema, uncond_y, guidance_scale),
            **self.kargs,
        )
        nfe = solution.nfev
        logging.info(f"sample nfe:{nfe}")
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(x.device).type(torch.float32)
        return x.detach()

    def _ode_func(self, t, x, y, shape, device, use_ema, uncond_y, guidance_scale):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        t = torch.ones(shape[0], device=device) * t
        if not self.model.continuous:
            t /= self.ns.N
        drift = self.model.rode(x, t, y, use_ema, uncond_y, guidance_scale)
        return to_flattened_numpy(drift)

    ####torch.sdeint   t_span should be increasing sequence, so  drift= -1 * drift
    def _sde_sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        shape = x.shape
        t_span = (self.ns.EPS, self.ns.T)
        dt = self.ns.T / self.sampling_steps
        if not self.model.continuous:
            t_span = (0, self.ns.N / self.ns.T)
            dt *= self.ns.N / self.ns.T
        sde = self._sde(x, y, use_ema, uncond_y, guidance_scale)
        ys = torchsde.sdeint(
            sde,
            x.reshape((shape[0], -1)),
            t_span,
            dt=dt,
            # adaptive=True,
            method=self.method.lower(),
            **self.kargs,
        )
        logging.info(f"sample nfe:{sde.nfe}")
        x = ys[-1, :].reshape(shape)
        return x

    def _sde(self, x, y, use_ema, uncond_y, guidance_scale):
        sampler = self
        shape = x.shape

        class _SDE:
            def __init__(self) -> None:
                self.diffusion = None
                self.noise_type = "diagonal"
                self.sde_type = "ito"
                self.nfe = 0

            def f(self, t, z):
                # logging.info(f"in f, t:{t}")
                t = torch.ones(shape[0], device=t.device) * t
                if sampler.model.continuous:
                    t = sampler.ns.T - t
                else:
                    t = (sampler.ns.N / sampler.ns.T - t) / sampler.ns.N
                # logging.info(f"in f, predict: t{t}")
                z = z.reshape(x.shape)
                drift, diffusion = sampler.model.rsde(z, t, y, use_ema, uncond_y, guidance_scale)
                self.nfe += 1
                self.diffusion = diffusion
                return -drift.reshape((shape[0], -1))

            def g(self, t, z):
                if self.diffusion is None:
                    logging.info(f"in g, t:{t}")
                    t = torch.ones(shape[0], device=t.device) * t
                    _, self.diffusion = sampler.model.rsde(z, t, y, use_ema, uncond_y, guidance_scale)
                    self.nfe += 1
                diffusion = einops.repeat(self.diffusion, "B -> B C H W", C=shape[1], H=shape[2], W=shape[3])
                self.diffusion = None
                return diffusion.reshape((shape[0], -1))

        return _SDE()
