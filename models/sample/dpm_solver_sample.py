from abc import abstractmethod

import numpy as np
import torch
import tqdm

from lib.tensor_trans import batch_scalar_prod
from optimizer.dpm_solver_pp import DPM_Solver, NoiseScheduleVP

from . import sample_factory
from .base_sample import SDESample


@sample_factory.register_sampler(name="dpm_solver")
class DPMSolverSample(SDESample):
    def __init__(self, model, sampling_method, sampling_steps=10) -> None:
        super().__init__(model, sampling_method)
        self.sampling_steps = sampling_steps

    def _sample(self, x, y, use_ema, uncond_y, guidance_scale, denoise=False):
        noise_schedule = self.get_noise_schedule()

        def model_fn(x, t_continuous):
            # t = t_continuous * self.N
            # logging.info(t[0])
            preds, _ = self.model.sampling_predict(x, t_continuous, y, use_ema, uncond_y, guidance_scale)
            return preds

        dmp_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        samples = dmp_solver.sample(x, steps=self.sampling_steps, eps=self.model.EPS, T=self.model.T)
        return samples

    def get_noise_schedule(self):
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.model.ns.betas)
        return noise_schedule
