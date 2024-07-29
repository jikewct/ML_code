# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from configs.config_utils import *
from configs.default_mnist_configs import get_default_configs


def get_config():
    config = get_default_configs()
    c(config, "training").update(
        batch_size=16,
        epochs=100,
        snapshot_freq=1000,
        log_freq=100,
        eval_freq=1000,
        test_metric_freq=50000,
        resume=True,
        # model_checkpoint="./data/checkpoints/generative_model/flowMatching/unet-cifar10-65000-model",
    )

    c(config, "model").update(
        name="flowMatching",
        nn_name="unet",
    )
    c(config, "model").update(
        channel_mults=(1, 2),
        activation="silu",
        base_channels=128,
        time_emb_dim=512,
        time_emb_scale=1.0,
        dropout=0.1,
        attention_resolutions=(1,),
        norm="gn",
        num_groups=32,
        num_res_blocks=2,
    )
    c(config, "model", "flowMatching").update(
        num_scales=1000,
    )

    c(config, "sampling").update(
        log_freq=1,
        method="rk45",
        sampling_steps=50,
        denoise=True,
    )
    c(config, "sampling", "rk45").update(
        rtol=1e-3,
        atol=1e-3,
    )

    config.pipeline = "FlowMatchingPipeLine"

    return config
