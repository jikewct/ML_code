from models import *

from .base_pipeline import BasePipeLine

__all__ = ["SMLDPipeLine", "VESDEPipeLine"]


class SMLDPipeLine(BasePipeLine):

    def __init__(self, config):
        super().__init__(config)


class VESDEPipeLine(SMLDPipeLine):

    def __init__(self, config):
        super().__init__(config)
