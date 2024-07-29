import ml_collections
import ml_collections.config_dict
import torch

from configs.config_utils import *


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    c(config, "training").update(
        batch_size=64,
        epochs=100,
        snapshot_freq=1000,
        log_freq=100,
        eval_freq=1000,
        test_metric_freq=50000,
        ## store additional checkpoints for preemption in cloud computing environments
        snapshot_freq_for_preemption=10000,
        ## produce samples at each snapshot.
        snapshot_sampling=True,
        likelihood_weighting=False,
        continuous=False,
        reduce_mean=False,
        model_checkpoint="",
        optim_checkpoint="",
        ckpt_dir="./data/checkpoints",
        log_to_wandb=True,
        project_name="generative_model",
        debug_groups=10,
        enable_debug=True,
    )
    # sampling
    c(config, "sampling").update(
        method="",
        enable_debug=False,
        # n_steps_each=1,
        # noise_removal=True,
        # probability_flow=False,
        # snr=0.16,
        # sample_steps=1000,
        log_freq=10,
    )
    # evaluation
    c(config, "eval").update(
        begin_ckpt=9,
        end_ckpt=26,
        batch_size=1024,
        enable_sampling=False,
        num_samples=50000,
        enable_loss=True,
        enable_bpd=False,
        bpd_dataset="test",
    )

    c(config, "test").update(
        batch_size=128,
        num_samples=500,
        save_path="./data/test/",
    )
    c(config, "fast_fid").update(
        begin_step=10000,
        end_step=1000000,
        num_samples=1000,
        batch_size=64,
        save_path="./data/test/",
        ds_state_file="/home/jikewct/public/jikewct/Dataset/cifar10/cifar-10-images/train_fid_stats.npz",
    )
    # data
    c(config, "data").update(
        dataset="cifar10",
        img_size=(32, 32),
        img_channels=3,
        num_classes=10,
        # img_class="",
        root_path="/home/jikewct/public/jikewct/Dataset/cifar10",
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
    )
    c(config.data, "cifar10").update()
    # model
    c(config, "model").update(
        name="",
        nn_name="",
        # activation="silu",
        # base_channels=128,
        # time_emb_dim=512,
        # time_emb_scale=1.0,
        # dropout=0.1,
        # attention_resolutions=(1,),
        # norm="gn",
        # num_groups=32,
        # channel_mults=(1, 2),
        # num_res_blocks=2,
        enable_ema=True,
        loss_type="l2",
        use_labels=False,
        # embedding_type="fourier",
        # ode=False,
        enable_lora=False,
        load_ckpt_strict=False,
        grad_checkpoint=False,
    )
    c(config.model, "ema").update(
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    )
    c(config.model, "lora").update(
        enable_lora=[True],
        lora_dim=4,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias_trainable="all",
    )
    # optimization
    c(config, "optim").update(
        decay_num=2,
        weight_decay=0.0,
        optimizer="Adam",
        lr=2e-4,
        beta1=0.9,
        eps=1e-8,
        warmup=5000,
        grad_clip=1.0,
        amsgrad=False,
    )
    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config.pipeline = ""
    return config
