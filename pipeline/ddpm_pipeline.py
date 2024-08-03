from models import *

from .base_pipeline import BasePipeLine

__all__ = ["DDPMPipeLine"]


class DDPMPipeLine(BasePipeLine):

    def __init__(self, config):
        super().__init__(config)
