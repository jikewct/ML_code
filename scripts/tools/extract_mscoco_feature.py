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
from configs.default_mscoco_configs import get_default_configs
from datasets import dataset_factory, dataset_utils
from models import FlowMatching, model_factory, model_utils
from network import autoencoder_kl, clip, net_factory
from network.layers import layer_utils


def extract_feature(argv):
    config = get_default_configs()
    c(config, "model").update(
        name="fm_ldm",
        nn_name="uvit",
        autoencoder_name="autoencoder_kl",
        grad_checkpoint=True,
    )
    c(config, "model", "autoencoder_kl").update(
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
        scale_factor=0.18215,
    )

    print(config.data)
    autoencoder = net_factory.create_network(config, "autoencoder_name").to(config.device)
    dataset = dataset_factory.create_dataset(config)
    print(dataset.states())
    textencoder = clip.FrozenCLIPEmbedder(version="/home/jikewct/public/jikewct/Model/clip-vit-large-patch14").to(config.device)
    autoencoder.eval()
    textencoder.eval()

    train_loader, val_loader = dataset.get_dataloader(1)

    with torch.no_grad():
        for mode, loader in zip(["train", "val"], [train_loader, val_loader]):
            save_path = os.path.join("/home/jikewct/Dataset/coco2017", "coco_256_feature_tmp", mode)
            os.makedirs(save_path, exist_ok=True)

            index = 0
            for x, captions in loader:
                if len((x.shape)) == 3:
                    x = x[None, ...]
                # print(x.shape)
                x = x.to(config.device)
                moments = autoencoder(x, fn="encode_moments").squeeze(0).detach().cpu().numpy()
                captions = [items[0] for items in captions]
                # print(captions)
                latent = textencoder.encode(captions).detach().cpu().numpy()
                np.savez(os.path.join(save_path, f"{index}.npz"), img=moments, text=latent)
                # print(moments.shape, latent.shape)
                # for i in range(len(latent)):
                #     z = latent[i].detach().cpu().numpy()
                #     np.save(os.path.join(save_path, f"{index}_{i}.npy"), z)
                index += 1
                if index % 1000 == 0:
                    print(f"processed {index} images in {mode} loader...")
                # if index >= 2:
                #     break


if __name__ == "__main__":
    app.run(extract_feature)
