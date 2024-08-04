from abc import abstractmethod

import numpy as np
import torch
import tqdm

from lib.tensor_trans import batch_scalar_prod

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="ode")
class ODESample(SDESample):
    def __init__(self, model, sampling_method, sampling_steps=10) -> None:
        super().__init__(model, sampling_method)
        self.sampling_steps = sampling_steps

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        dt = -self.ns.T / self.sampling_steps
        if not self.model.continuous:
            dt *= self.ns.N
        steps = np.linspace(0, self.ns.N - 1, self.sampling_steps).round().astype(np.int32)
        for i in tqdm.tqdm(steps, desc="ode sample", total=self.sampling_steps, leave=False):
            ## t from T to eps
            t = (self.ns.EPS - self.ns.T) * i / self.ns.N + self.ns.T
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift = self.model.rode(x, vec_t, y, use_ema, uncond_y, guidance_scale)
            x = x + drift * dt
            # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x
