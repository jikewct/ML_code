import logging
import os
import sys
from copy import deepcopy
from functools import partial

import imageio
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
from network import *


def extract_feature(argv):
    config = get_default_configs()
    c(config, "model").update(
        name="fm_ldm",
        nn_name="uvit",
        grad_checkpoint=True,
    )
    c(config, "model", "fm_ldm").update(
        autoencoder_name="frozen_autoencoder_kl",
    )
    c(config, "model", "fm_ldm", "frozen_autoencoder_kl").update(
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
    autoencoder = net_factory.create_network(config.model[config.model.name], "autoencoder_name").to(config.device)
    dataset = dataset_factory.create_dataset(config)
    print(dataset.states())
    textencoder = FrozenCLIPEmbedder(pretrained_path="/home/jikewct/public/jikewct/Model/clip-vit-large-patch14").to(config.device)
    autoencoder.eval()
    textencoder.eval()
    train_loader, val_loader = dataset.get_dataloader(1)
    empty_caption = [""]
    empty_latent = textencoder.encode(empty_caption).detach().cpu().numpy()
    empty_save_path = os.path.join("/home/jikewct/Dataset/coco2017", "coco_256_feature_tmp")
    os.makedirs(empty_save_path, exist_ok=True)
    np.save(os.path.join(empty_save_path, "empty_latent.npy"), empty_latent)
    print(empty_latent.shape)
    print(empty_latent)
    return
    with torch.no_grad():
        for mode, loader in zip(["train", "val"], [train_loader, val_loader]):
            save_path = os.path.join("/home/jikewct/Dataset/coco2017", "coco_256_feature_tmp", mode)
            os.makedirs(save_path, exist_ok=True)

            index = 0
            for x, captions in loader:
                if len((x.shape)) == 3:
                    x = x[None, ...]
                print(x.shape)
                x = x.to(config.device)
                moments = autoencoder(x, fn="encode_moments").squeeze(0).detach().cpu().numpy()
                captions = [items[0] for items in captions]
                print(captions)
                x = ((x + 1) * 0.5).squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
                x = (x * 255).round().astype(np.uint8)
                print(x.shape)
                imageio.imsave(os.path.join(save_path, f"{index}.jpg"), x)
                with open(os.path.join(save_path, f"{index}.txt"), "w") as file:
                    file.write("\n".join(captions))
                latent = textencoder.encode(captions).detach().cpu().numpy()
                np.savez(os.path.join(save_path, f"{index}.npz"), img=moments, text=latent)
                # print(moments.shape, latent.shape)
                # for i in range(len(latent)):
                #     z = latent[i].detach().cpu().numpy()
                #     np.save(os.path.join(save_path, f"{index}_{i}.npy"), z)
                index += 1
                if index % 1000 == 0:
                    print(f"processed {index} images in {mode} loader...")
                if index >= 2:
                    break


if __name__ == "__main__":
    app.run(extract_feature)
