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

# Lint as: python3
"""Train the original DDPM model."""

from configs.config_utils import *
from configs.default_afhq_configs import get_default_configs


def get_config():
    config = get_default_configs()
    c(config, "training").update(
        batch_size=64,
        epochs=10000,
        snapshot_freq=500,
        log_freq=50,
        eval_freq=500,
        test_metric_freq=50000,
        resume=True,
        resume_path="",
        resume_step=-1,
        # model_checkpoint="./data/checkpoints/generative_model/flowMatching/uvit/afhq/96X96/30-network.pth",
    )
    c(config, "data").update(
        img_size=(96, 96),
    )
    c(config.data, "afhq").update(
        img_class="cat",
    )
    c(config, "model").update(
        name="flowMatching",
        nn_name="uvit",
        grad_checkpoint=True,
    )
    c(config.model, "flowMatching").update()
    c(config.model, "uvit").update(
        img_size=config.data.img_size[0],
        patch_size=8,
        embed_dim=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
        use_checkpoint=config.model.grad_checkpoint,
    )

    c(config, "test").update(
        save_path="./data/test/",
        batch_size=64,
        num_samples=100,
    )

    c(config, "sampling").update(
        log_freq=1,
        method="rk45",
        sampling_steps=50,
        denoise=True,
    )
    c(config.sampling, "rk45").update(
        rtol=1e-3,
        atol=1e-3,
    )
    c(config.sampling, "ode").update()

    c(config, "optim").update(
        optimizer="adamw",
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )
    c(config, "lr_scheduler").update(
        name="customized",
        warmup_steps=2000,
    )
    config.pipeline = "FlowMatchingPipeLine"

    return config
