from abc import abstractmethod

import torch

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import interpolate_fn

from . import noise_schedule_factory
from .base_noise_schedule import BaseNoiseSchedule


@noise_schedule_factory.register_noise_scheduler(name="rfns")
class RectifiedNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, continuous, device) -> None:
        super().__init__(continuous, device)

    # def get_alpha(self, t):
    #     return self.T - t

    def marginal_coef(self, t):

        return torch.ones_like(t), torch.ones_like(t)

    # def marginal_prob(self, x, t):
    #     mean_coeff, std = self.marginal_coef(t)
    #     mean = batch_scalar_prod(mean_coeff, x)
    #     return mean, std
