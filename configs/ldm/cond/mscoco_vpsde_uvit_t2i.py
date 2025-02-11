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
from configs.default_mscoco_configs import get_default_configs


def get_config():
    config = get_default_configs()
    c(config, "training").update(
        batch_size=64,
        epochs=10000,
        snapshot_freq=500,
        log_freq=50,
        eval_freq=50,
        test_metric_freq=1000000,
        resume=True,
        continuous=False,
        # resume_path="/home/jikewct/public/jikewct/Repos/ml_code/data/checkpoints/generative_model/fm_ldm/uvit/afhq_32x32_feature/32X32",
        # model_checkpoint="/home/jikewct/public/jikewct/Model/uvit/mscoco_uvit_small.pth",
    )
    c(config, "data").update(
        dataset="mscoco_32x32_feature",
        img_size=(32, 32),
        img_channels=4,
        root_path="/home/jikewct/Dataset/coco2017/coco_256_feature",
    )
    c(config, "data", "mscoco_32x32_feature").update()
    c(config, "model").update(
        name="vpsde_ldm",
        nn_name="uvit_t2i",
        conditional=True,
        grad_checkpoint=True,
    )
    c(config, "model", "condition").update(
        condition_type="text",
        cond_embedder_name="frozen_clip_embedder",
        cfg=True,
        p_cond=0.1,
        empty_latent_path="/home/jikewct/Dataset/coco2017/coco_256_feature/empty_latent.npy",
    )
    c(config, "model", "ldm").update(
        autoencoder_name="frozen_autoencoder_kl",
    )

    c(config, "model", "vpsde").update(
        scheduler="vpns",  ## vpns,vens, rfns
    )
    c(config, "model", "vpns").update(
        schedule_type="sd",  ## sd, linear
        num_scales=1000,
        std_min=0.00085,
        std_max=0.0120,
    )
    c(config, "model", "vpsde_ldm").update()
    c(config, "model", "condition", "frozen_clip_embedder").update(
        pretrained_path="/home/jikewct/public/jikewct/Model/clip-vit-large-patch14",
    )

    c(config, "model", "uvit_t2i").update(
        # cacl from autoencoder_kl ch_mult config
        img_size=config.data.img_size[0],
        patch_size=2,
        in_chans=config.data.img_channels,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77,
        use_checkpoint=config.model.grad_checkpoint,
    )
    c(config, "model", "ldm", "frozen_autoencoder_kl").update(
        double_z=True,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
        embed_dim=4,
        pretrained_path="/home/jikewct/public/jikewct/Model/stable_diffusion/stable-diffusion/autoencoder_kl.pth",
        scale_factor=0.23010,
    )
    c(config, "test").update(
        batch_size=5,
        num_samples=1,
        save_path="./data/test/",
    )
    c(config, "sampling").update(
        log_freq=1,
        method="pc",  # ode, numerical, pc, dpm_solver
        denoise=True,
        sampling_conditions=[
            "A green train is coming down the tracks.",
            "A group of skiers are preparing to ski down a mountain.",
            "A small kitchen with a low ceiling.",
            "A group of elephants walking in muddy water.",
            "A living area with a television and a table.",
        ],
        guidance_scale=1.0,
    )
    c(config, "sampling", "numerical").update(
        rtol=1e-3,
        atol=1e-3,
        equation_type="sde",
        # ode in (eluer, rk45),
        # sde in 'euler', 'euler_heun', 'heun', 'log_ode', 'midpoint', 'milstein', 'reversible_heun', 'srk']
        method="srk",
        sampling_steps=20,
    )
    c(config, "sampling", "ode").update(
        sampling_steps=50,
    )
    c(config, "sampling", "dpm_solver").update(
        sampling_steps=50,
    )
    c(config, "sampling", "pc").update(
        predictor="ancestralsampling",  # euler,reversediffusion,ancestralsampling,""
        corrector="langevin",  # ald,langevin, ""
        n_step_each=1,
        snr=0.16,
    )
    c(config, "optim").update(
        optimizer="adamw",
    )
    c(config, "optim", "adamw").update(
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )
    c(config, "lr_scheduler").update(
        name="customized",
    )
    c(config, "lr_scheduler", "customized").update(
        warmup_steps=2000,
    )

    config.pipeline = "LDMPipeLine"

    return config
