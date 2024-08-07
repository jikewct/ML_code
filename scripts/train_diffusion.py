import datetime
import logging
import logging.config
from pathlib import Path

import numpy as np
import torch
import torch.utils
import wandb

from pipeline import *

__all__ = ["TrainDiffusion"]


class TrainDiffusion:

    def __init__(self, args, config):
        self.args = args
        self.config = config

    def init_wandb(self, config, pipeline: BasePipeLine):
        if config.training.log_to_wandb:
            if config.training.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")
            wandb.require("core")
            run = wandb.init(
                dir="./data/",
                project=config.training.project_name,
                config=vars(config),
                resume="allow",
                id=pipeline.uuid,
                name=pipeline.Name + "-" + self.args.mode + datetime.datetime.now().strftime(":%Y-%m-%d-%H-%M"),
            )
            # wandb.log({"net": pipeline.get_model()})
            logging.info(f"wandb url:{run.get_url()}")
            return run

    def log_runner_status(self):
        project_name = self.config.training.project_name
        model_name = self.config.model.name
        net_name = self.config.model.nn_name
        dataset = self.config.data.dataset
        mode = self.args.mode
        logging.info(f"{project_name}-{model_name}-{net_name}-{dataset}:{mode}")

    def pre_set_config(self):
        if self.args.mode == "train":
            self.config.sampling.enable_debug = True
            self.config.training.log_to_wandb = True
        elif self.args.mode == "debug_sampling":
            self.config.sampling.enable_debug = True
            self.config.training.log_to_wandb = True

    def run(self):
        wandb_run = None
        try:
            self.pre_set_config()
            pipeline = eval(self.config.pipeline)(self.config)
            # ddpmPipeLine = DDPM_pipeline.DDPMPipeLine(self.config)
            self.log_runner_status()
            if self.args.mode == "train":
                wandb_run = self.init_wandb(self.config, pipeline)
                pipeline.train_loop()
            elif self.args.mode == "eval":
                pipeline.eval_loop()
            elif self.args.mode == "test":
                pipeline.test_loop()
            elif self.args.mode == "debug_sampling":
                wandb_run = self.init_wandb(self.config, pipeline)
                pipeline.debug_sampling()
        except KeyboardInterrupt:
            logging.info("key board interrupt, run finished early")
        except Exception:
            logging.info("exception, run finished early", exc_info=True)
        finally:
            if wandb_run is not None:
                wandb_run.finish(quiet=True)
