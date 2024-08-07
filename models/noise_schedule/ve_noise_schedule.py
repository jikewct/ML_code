import logging
import math
from abc import abstractmethod

import torch

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import interpolate_fn

from . import noise_schedule_factory
from .guassian_noise_schedule import GuassianNoiseSchedule


class VENoiseSchedule(GuassianNoiseSchedule):
    def __init__(self, continuous, device, std_min, std_max, num_scales, schedule_type) -> None:
        super().__init__(continuous, device, std_min, std_max, num_scales, schedule_type)
        self.init_discrete_parameter()

    def init_discrete_parameter(self):
        if not self.continuous:
            _marginal_sigmas = self._generate_discrete_marginal_sigmas()
            _marginal_sigmas_prev = torch.concat((torch.tensor([0.0], device=self.device), _marginal_sigmas[:-1]))
            self.recursive_cond_sigmas = torch.sqrt(_marginal_sigmas**2 - _marginal_sigmas_prev**2).reshape((1, -1))
            self.discrete_t = torch.linspace(self.EPS, self.T, self.N, device=self.device).reshape((1, -1))

    @abstractmethod
    def _generate_discrete_marginal_sigmas(self):
        pass

    @abstractmethod
    def _get_sigma(self, t):
        pass

    @abstractmethod
    def _get_continuous_recursive_cond_sigma(self, t):
        pass

    def marginal_coef(self, t):
        std = self._get_sigma(t)
        return torch.ones_like(t), std

    def recursive_cond_coef(self, t):
        if self.continuous:
            std = self._get_continuous_recursive_cond_sigma(t)
        else:
            std = interpolate_fn(t.reshape((-1, 1)), self.discrete_t.clone(), self.recursive_cond_sigmas.clone()).reshape((-1,))
        return torch.ones_like(t), std


@noise_schedule_factory.register_noise_scheduler(name="linear_vens")
class VELinearNoiseSchedule(VENoiseSchedule):
    def __init__(self, **kargs) -> None:
        print(kargs)
        super().__init__(**kargs)

    def _generate_discrete_marginal_sigmas(self):
        return torch.linspace(self.std_min, self.std_max, self.N).to(self.device)

    def _get_sigma(self, t):
        return (self.std_max - self.std_min) * t + self.std_min

    def _get_continuous_recursive_cond_sigma(self, t):
        return torch.sqrt(2 * (self.std_max - self.std_min) * self._get_sigma(t))


@noise_schedule_factory.register_noise_scheduler(name="geo_vens")
class VEGeometricNoiseSchedule(VENoiseSchedule):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

    def _generate_discrete_marginal_sigmas(self):
        return torch.exp(torch.linspace(math.log(self.std_min), math.log(self.std_max), self.N)).to(self.device)

    def _get_sigma(self, t):
        return self.std_min * torch.pow(self.std_max / self.std_min, t)

    def _get_continuous_recursive_cond_sigma(self, t):
        return math.sqrt(2 * math.log(self.std_max / self.std_min)) * self._get_sigma(t)
