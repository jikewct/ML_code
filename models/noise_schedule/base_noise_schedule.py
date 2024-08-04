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
        return 999

    @abstractmethod
    def marginal_coef(self, t):
        pass

    # def get_alpha(self, t):
    #     return torch.ones_like(t)

    # def get_alpha_cum(self, t):
    #     return torch.ones_like(t)

    # @abstractmethod
    # def get_beta(self, t):
    #     pass

    def states(self):
        state = {"ns_scheduler": self.__class__.__name__}
        state.update(fileter_object_states(self))
        return state
