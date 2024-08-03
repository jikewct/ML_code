import logging
import os
import sys
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from absl import app

sys.path.append(".")
sys.path.append("..")
os.chdir("/home/jikewct/public/jikewct/Repos/ml_code")
from configs.config_utils import c
from configs.ldm.cond.mscoco_vpsde_uvit_t2i import get_config


def test_config(argv):
    config = get_config()
    print(config)


if __name__ == "__main__":
    app.run(test_config)
