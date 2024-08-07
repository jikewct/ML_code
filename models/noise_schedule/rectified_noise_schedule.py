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

    @property
    def EPS(self):
        return 0.0

    def marginal_coef(self, t):
        return torch.ones_like(t), torch.ones_like(t)

    def recursive_cond_coef(self, t):
        return torch.ones_like(t), torch.ones_like(t)
