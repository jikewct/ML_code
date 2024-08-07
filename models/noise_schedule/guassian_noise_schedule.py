from abc import abstractmethod

import torch

from lib.tensor_trans import *
from optimizer.dpm_solver_pp import interpolate_fn
from utils.monitor import fileter_object_states

from .base_noise_schedule import BaseNoiseSchedule


class GuassianNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, continuous, device, std_min, std_max, num_scales, schedule_type) -> None:
        super().__init__(continuous, device)
        self.std_min = std_min
        self.std_max = std_max
        self.schedule_type = schedule_type
        self.num_scales = num_scales

    @property
    def N(self):
        return self.num_scales
