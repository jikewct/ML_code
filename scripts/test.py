import logging
import os

import numpy as np
import wandb
from absl import app, flags
from ml_collections.config_flags import config_flags
from pipeline.datasets import data_trans

__all__ = ["Test"]


class Test:

    def __init__(self, args, config):
        logging.info(args.mode)
        #logging.info(config)

    def test_wandb(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project="test")

        x = np.linspace(0, 100, 100)
        log_info = {'x': x, 'y': x}
        wandb.log(log_info)
        wandb.finish()

    def run(self):
        logging.info("test running....")
        self.test_wandb()
