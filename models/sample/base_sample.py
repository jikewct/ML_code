import logging
from abc import abstractmethod

import numpy as np
import torch
import tqdm

from lib.tensor_trans import batch_scalar_prod
from models.base_model import BaseModel
from models.sde import SDE
from utils.monitor import fileter_object_states


class BaseSample:
    def __init__(self, model: BaseModel):
        self.model = model
        self.ns = model.ns

    def sample(self, y, use_ema, uncond_y, guidance_scale, denoise=False):
        # logging.info(self.states())
        return self._sample(y, use_ema, uncond_y, guidance_scale, denoise)

    @abstractmethod
    def _sample(self, y, use_ema, uncond_y, guidance_scale, denoise=False):
        pass

    def states(self):
        state = {"sampler": self.__class__.__name__}
        state.update(fileter_object_states(self))
        state.update(self.ns.states())
        return state


class SDESample(BaseSample):
    def __init__(self, model: SDE):
        super().__init__(model)
        self.model = model
