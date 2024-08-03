from abc import abstractmethod

import torch

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import interpolate_fn

from .base_noise_schedule import BaseNoiseSchedule


class VPNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, continuous, device, std_min, std_max, num_scales, schedule_type) -> None:
        super().__init__(continuous, device, std_min, std_max, num_scales, schedule_type)
        self.init_discrete_parameter()

    def init_discrete_parameter(self):
        if not self.continous:
            self.betas = self.generate_discrete_betas()
            self.log_alpha_cums = 0.5 * torch.log(1 - self.betas).cumsum(dim=0).reshape((1, -1))
            self.discrete_t = torch.linspace(self.EPS, self.T, self.N, device=self.device).reshape((1, -1))

    @abstractmethod
    def generate_discrete_betas():
        pass

    @abstractmethod
    def get_continous_alpha_cum(self, t):
        pass

    def get_alpha(self, t):
        return 1.0 - self.get_beta(t)

    def get_alpha_cum(self, t):
        if self.continous:
            return self.get_continous_alpha_cum(t)
        log_alpha_cum = interpolate_fn(t.reshape((-1, 1)), self.discrete_t.clone(), self.log_alpha_cums.clone()).reshape((-1,))
        return torch.exp(log_alpha_cum) ** 2

    def marginal_std(self, t):
        return self.marginal_coef(t)[1]

    def marginal_coef(self, t):
        alpha_cum_t = self.get_alpha_cum(t)
        mean_coeff = torch.sqrt(alpha_cum_t)
        std = torch.sqrt(1.0 - mean_coeff**2)
        return mean_coeff, std

    def marginal_prob(self, x, t):
        mean_coeff, std = self.marginal_coef(t)
        mean = batch_scalar_prod(mean_coeff, x)
        return mean, std


class VPLinearNoiseSchedule(VPNoiseSchedule):
    def __init__(self, **kargs) -> None:
        print(kargs)
        super().__init__(**kargs)

    def generate_discrete_betas(self):
        return torch.linspace(self.std_min, self.std_max, self.N).to(self.device)

    def get_beta(self, t):
        return (self.beta_max - self.beta_min) * t + self.beta_min

    def get_continous_alpha_cum(self, t):
        return torch.exp(-0.5 * (2 * t**2 * (self.beta_max - self.beta_min) + t * self.beta_min))


class VPSDNoiseSchedule(VPNoiseSchedule):
    def __init__(self, *args, **kargs) -> None:
        super().__init__(*args, **kargs)

    def generate_discrete_betas(self):
        return (torch.linspace(self.std_min**0.5, self.std_max**0.5, self.N) ** 2).to(self.device)

    def get_beta(self, t):
        return ((self.std_max**0.5 - self.std_min**0.5) * t + self.std_min**0.5) ** 2

    def get_continous_alpha_cum(self, t):
        a = self.std_max**0.5 - self.std_min**0.5
        b = self.std_min**0.5
        return torch.exp(-0.5 * (1.0 / 3 * t**3 * a**2 + a * b * t ** +(b**2) * t))

    # def generate_cosine_schedule(self, T, s=0.008):

    #     def f(t, T):
    #         return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

    #     alphas = []
    #     f0 = f(0, T)

    #     for t in range(T + 1):
    #         alphas.append(f(t, T) / f0)
    #     betas = []

    #     for t in range(1, T + 1):
    #         betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))

    #     return np.array(betas)
