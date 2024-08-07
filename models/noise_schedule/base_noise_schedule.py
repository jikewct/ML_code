from abc import abstractmethod

import torch

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import interpolate_fn
from utils.monitor import fileter_object_states


class BaseNoiseSchedule:
    def __init__(self, continuous, device) -> None:
        self.continuous = continuous
        self.device = device

    @property
    def T(self):
        return 1.0

    @property
    def EPS(self):
        return 1e-5

    @property
    def N(self):
        return 1000

    @abstractmethod
    def marginal_coef(self, t):
        pass

    @abstractmethod
    def recursive_cond_coef(self, t):
        pass

    def generate_rand_t(self, batch_size):
        return torch.rand((batch_size,), device=self.device) * (self.T - self.EPS) + self.EPS

    def states(self):
        state = {"ns_scheduler": self.__class__.__name__}
        state.update(fileter_object_states(self))
        return state
