import logging
from abc import abstractmethod

import numpy as np
import torch
import tqdm
from scipy import integrate

from lib.tensor_trans import *

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="rk45")
class RK45Sample(SDESample):
    def __init__(self, model, sampling_method, **kargs) -> None:
        super().__init__(model, sampling_method)
        self.kargs = kargs

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        shape = x.shape
        t_span = (self.ns.T, self.ns.EPS)
        if not self.model.continuous:
            t_span = (self.ns.N, 0)
        solution = integrate.solve_ivp(
            self.ode_func,
            t_span,
            to_flattened_numpy(x),
            method="RK45",
            args=(y, shape, x.device, use_ema, uncond_y, guidance_scale),
            **self.kargs,
        )
        nfe = solution.nfev
        logging.info(f"sample nfe:{nfe}")
        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(x.device).type(torch.float32)
        return x.detach()

    def ode_func(self, t, x, y, shape, device, use_ema, uncond_y, guidance_scale):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        t = torch.ones(shape[0], device=device) * t
        if not self.model.continuous:
            t /= self.ns.N
        drift = self.model.rode(x, t, y, use_ema, uncond_y, guidance_scale)
        return to_flattened_numpy(drift)
