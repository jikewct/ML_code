import ml_collections
import ml_collections.config_dict
import torch

from configs.config_utils import *


def default_configs():
    defualt_config = ml_collections.ConfigDict()
    # training
    n(defualt_config, "training").update(
        batch_size=64,
        epochs=100,
        snapshot_freq=1000,
        log_freq=100,
        eval_freq=1000,
        test_metric_freq=50000,
        likelihood_weighting=False,
        continuous=False,
        model_checkpoint="",
        ckpt_dir="./data/checkpoints",
        log_to_wandb=True,
        project_name="generative_model",
        debug_groups=10,
        enable_debug=True,
        resume=False,
        resume_path="",
        resume_step=-1,
    )
    # sampling
    n(defualt_config, "sampling").update(
        method="",  # ode, numerical, pc, dpm_solver
        enable_debug=False,
        log_freq=10,
        sampling_conditions=None,  # None or List()
        guidance_scale=1.0,
    )
    n(defualt_config, "sampling", "ode").update(
        sampling_steps=50,
    )
    n(defualt_config, "sampling", "numerical").update(
        rtol=1e-3,
        atol=1e-3,
        equation_type="sde",  ## ode, sde
        method="rk45",  ## ode in (eluer, rk45), sde in (eluer, srk)
        sampling_steps=50,
    )
    n(defualt_config, "sampling", "pc").update(
        predictor="",  # euler,reversediffusion,ancestralsampling,""
        corrector="",  # ald,langevin, ""
        n_step_each=1,
        snr=0.16,
    )
    n(defualt_config, "sampling", "dpm_solver").update(
        sampling_steps=50,
    )
    # evaluation
    n(defualt_config, "eval").update(
        begin_ckpt=9,
        end_ckpt=26,
        batch_size=1024,
        enable_sampling=False,
        num_samples=50000,
        enable_loss=True,
        enable_bpd=False,
        bpd_dataset="test",
    )

    n(defualt_config, "test").update(
        batch_size=128,
        num_samples=500,
        save_path="./data/test/",
    )
    n(defualt_config, "fast_fid").update(
        begin_step=10000,
        end_step=1000000,
        num_samples=1000,
        batch_size=64,
        save_path="./data/test/",
        ds_state_file="/home/jikewct/public/jikewct/Dataset/cifar10/cifar-10-images/train_fid_stats.npz",
    )
    # data

    n(defualt_config, "data").update(
        dataset="mnist",
        img_size=(28, 28),
        img_channels=1,
        num_classes=10,
        # img_class="",
        root_path="/home/jikewct/public/jikewct/Dataset/mnist",
        random_flip=True,
        centered=True,
        uniform_dequantization=False,
    )
    n(defualt_config, "data", "mnist").update()
    n(defualt_config, "data", "cifar10").update()
    n(defualt_config, "data", "lsun").update()
    n(defualt_config, "data", "afhq").update(
        # all or one of (cat|dog|wild)
        img_class="cat"
    )
    n(defualt_config, "data", "mscoco").update(
        year="2017",
    )
    n(defualt_config, "data", "mscoco_32x32_feature").update()
    n(defualt_config, "data", "afhq_32x32_feature").update()

    # model
    n(defualt_config, "model").update(
        name="",
        nn_name="",
        load_ckpt_strict=False,
        grad_checkpoint=True,
        enable_ema=True,
        loss_type="l2",
        ###network config
        predict_type="noise",
        conditional=False,
        enable_lora=False,
    )
    # model condition
    n(defualt_config, "model", "condition").update(
        cfg=True,
        p_cond=0.1,
        condition_type="class",  ###class or text
        cond_embedder_name="frozen_clip_embedder",
        empty_latent_path="",
        empty_label=-1,
    )
    n(defualt_config, "model", "condition", "frozen_clip_embedder").update(
        pretrained_path="/home/jikewct/public/jikewct/Model/clip-vit-large-patch14",
    )

    # model alg
    n(defualt_config, "model", "flowMatching").update(
        scheduler="rfns",  ## vpns,vens, rfns
    )

    n(defualt_config, "model", "ddpm").update()
    n(defualt_config, "model", "ddim").update()
    n(defualt_config, "model", "ldm").update(
        autoencoder_name="frozen_autoencoder_kl",
    )
    n(defualt_config, "model", "vpsde").update(
        scheduler="vpns",  ## vpns,vens, rfns
    )
    n(defualt_config, "model", "vesde").update(
        scheduler="vens",  ## vpns,vens, rfns
    )
    n(defualt_config, "model", "rfns").update()

    n(defualt_config, "model", "fm_ldm").update(
        scheduler="rfns",  ## vpns,vens, rfns
    )
    n(defualt_config, "model", "vpsde_ldm").update(
        # autoencoder_name="frozen_autoencoder_kl",
        scheduler="vpns",  ## vpns,vens, rfns
    )
    n(defualt_config, "model", "ldm", "frozen_autoencoder_kl").update(
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
    n(defualt_config, "model", "uvit").update(
        img_size=defualt_config.data.img_size[0],
        patch_size=2,
        in_chans=defualt_config.data.img_channels,
        embed_dim=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
        use_checkpoint=defualt_config.model.grad_checkpoint,
    )

    n(defualt_config, "model", "uvit_t2i").update(
        # cacl from autoencoder_kl ch_mult config
        img_size=defualt_config.data.img_size[0],
        patch_size=2,
        in_chans=defualt_config.data.img_channels,
        embed_dim=512,
        depth=16,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77,
        use_checkpoint=defualt_config.model.grad_checkpoint,
    )

    n(defualt_config, "model", "ema").update(
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    )
    n(defualt_config, "model", "lora").update(
        enable_lora=[True],
        lora_dim=4,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias_trainable="all",
    )

    # model noise scheduler
    n(defualt_config, "model", "vpns").update(
        schedule_type="sd",  ## sd, linear
        num_scales=1000,
        std_min=0.00085,
        std_max=0.0120,
    )
    n(defualt_config, "model", "vens").update(
        schedule_type="sd",  ## sd, linear
        num_scales=1000,
        std_min=0.00085,
        std_max=0.0120,
    )
    n(defualt_config, "optim").update(
        optimizer="adamw",
    )
    n(defualt_config, "optim", "adamw").update(
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )
    n(defualt_config, "optim", "adam").update(
        decay_num=2,
        weight_decay=0.1,
        lr=2e-4,
        beta1=0.9,
        eps=1e-8,
        warmup=5000,
        grad_clip=1.0,
        amsgrad=False,
    )

    n(defualt_config, "lr_scheduler").update(
        name="customized",
        # warmup_steps=5000,
    )
    n(defualt_config, "lr_scheduler", "customized").update(
        warmup_steps=5000,
    )

    defualt_config.seed = 1234
    defualt_config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    defualt_config.pipeline = ""
    return defualt_config


DEFAULT_CONFIGS = default_configs()
