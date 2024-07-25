import ml_collections
import ml_collections.config_dict
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 64
    training.epochs = 100

    training.snapshot_freq = 4000
    training.log_freq = 100
    training.eval_freq = 4000
    training.test_metric_freq = 50000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = False
    training.reduce_mean = False
    training.model_checkpoint = ""
    training.optim_checkpoint = ""
    training.ckpt_dir = "./data/checkpoints"
    training.log_to_wandb = True
    training.project_name = "generative_model"
    training.debug_groups = 10
    training.enable_debug = True

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.sample_steps = 1000
    sampling.enable_debug = False
    sampling.log_freq = 10

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"

    config.test = test = ml_collections.ConfigDict()

    test.batch_size = 128
    test.num_samples = 500
    test.save_path = "./data/test/"

    config.fast_fid = fast_fid = ml_collections.ConfigDict()
    fast_fid.begin_step = 10000
    fast_fid.end_step = 1000000
    fast_fid.num_samples = 1000
    fast_fid.batch_size = 64
    fast_fid.save_path = "./data/test/"
    fast_fid.ds_state_file = "/home/jikewct/public/jikewct/Dataset/cifar10/cifar-10-images/train_fid_stats.npz"

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "afhq"
    data.img_size = (256, 256)
    data.img_channels = 3
    data.num_classes = 10
    data.root_path = "/home/jikewct/public/jikewct/Dataset/afhq/"
    data.img_class = "all"
    data.random_flip = True
    data.centered = True
    data.uniform_dequantization = False

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = ""
    model.nn_name = ""
    model.activation = "silu"
    model.base_channels = 128
    model.time_emb_dim = 512
    model.time_emb_scale = 1.0
    model.dropout = 0.1
    model.attention_resolutions = (1,)
    model.norm = "gn"
    model.num_groups = 32
    model.channel_mults = (1, 2)
    model.num_res_blocks = 2

    model.ema_decay = 0.9999
    model.ema_start = 5000
    model.ema_update_rate = 1
    model.enable_ema = True

    model.loss_type = "l2"
    model.use_labels = False

    model.lora_dim = 4
    model.lora_alpha = 4
    model.lora_dropout = 0.0
    model.enable_lora = [False]
    model.lora_bias_trainable = "all"
    model.embedding_type = "fourier"
    model.ode = False
    model.load_ckpt_strict = False
    model.grad_checkpoint = False

    # optimization
    config.optim = optim = ml_collections.ConfigDict()

    optim.decay_num = 2
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0
    optim.amsgrad = False

    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config.pipeline = ""
    return config
