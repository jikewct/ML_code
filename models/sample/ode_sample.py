from abc import abstractmethod

import numpy as np
import torch
import tqdm

from lib.tensor_trans import batch_scalar_prod

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="ode")
class ODESample(SDESample):
    def __init__(self, model, sampling_steps=10) -> None:
        super().__init__(model)
        self.sampling_steps = sampling_steps

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        rsde = self.model.reverse(probability_flow=True)
        dt = -self.model.T / self.sampling_steps * self.model.N
        steps = np.linspace(0, self.model.N - 1, self.sampling_steps).round().astype(np.int32)
        for i in tqdm.tqdm(steps, desc="ode sample", total=self.sampling_steps, leave=False):
            t = (self.model.EPS - self.model.T) * i / self.model.N + self.model.T
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift, _ = rsde.sde(x, vec_t, y, use_ema, uncond_y, guidance_scale)
            x = x + drift * dt
            # x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x
