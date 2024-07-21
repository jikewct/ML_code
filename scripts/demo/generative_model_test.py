import logging
import logging.config
import os

import diffusers
import numpy as np
import tqdm
from diffusers import DiffusionPipeline


def test_score_sde_ve():
    logging.config.fileConfig("./configs/conf/log.conf")
    batch_size = 1
    model_id = "E:\jikewct\Model\google\\ncsnpp-ffhq-1024"
    sde_ve = DiffusionPipeline.from_pretrained(model_id).to("cuda")
    images = sde_ve(batch_size=batch_size, num_inference_steps=100)["images"]
    filepath = "{model_id}/score_ve_generated_images/"
    os.mkdirs(os.path.split(filepath)[0], exist_ok=True)
    for i in range(0, len(images)):
        images[i].save("{}/score_ve_generated_images/{:05d}.png".format(model_id, batch_size + i))


def test_ddpm():
    NFE = 1000
    sample_num = 50000
    batch_size = 128
    model_id = "E:\jikewct\Model\google\\ddpm_ema_cifar10"
    ddpm = diffusers.DDPMPipeline.from_pretrained(model_id).to("cuda")
    ddpm.set_progress_bar_config(leave=False)
    batch_size_list = np.concatenate([[batch_size] * (sample_num // batch_size), [sample_num % batch_size]])
    for iter, num in tqdm.tqdm(enumerate(batch_size_list), desc="Processing", total=len(batch_size_list)):
        images = ddpm(batch_size=num, num_inference_steps=NFE)["images"]
        for i in range(0, len(images)):
            images[i].save("{}/ddpm_generated_images/{:05d}.png".format(model_id, iter * batch_size + i))


def test_ddim():
    NFE = 100
    sample_num = 50000
    batch_size = 128
    model_id = "E:\jikewct\Model\google\\ddpm_ema_cifar10"
    ddpm = diffusers.DDIMPipeline.from_pretrained(model_id).to("cuda")
    ddpm.set_progress_bar_config(leave=False)
    batch_size_list = np.concatenate([[batch_size] * (sample_num // batch_size), [sample_num % batch_size]])
    for iter, num in tqdm.tqdm(enumerate(batch_size_list), desc="Processing", total=len(batch_size_list)):
        images = ddpm(batch_size=num, num_inference_steps=NFE)["images"]
        for i in range(0, len(images)):
            images[i].save("{}/ddim_generated_images/{:05d}.png".format(model_id, iter * batch_size + i))


if __name__ == "__main__":
    # test_ddpm()
    test_score_sde_ve()
