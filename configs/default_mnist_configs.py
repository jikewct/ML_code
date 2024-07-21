import ml_collections
import ml_collections.config_dict
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128
    training.epochs = 100
    training.log_freq = 100
    training.eval_freq = 1000
    training.snapshot_freq = 1000
    training.test_metric_freq = 40000

    ## store additional checkpoints for preemption in cloud computing environments
    ## produce samples at each snapshot.
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.model_checkpoint = ""
    training.optim_checkpoint = ""
    training.ckpt_dir = "./data/checkpoints"
    training.log_to_wandb = False
    training.project_name = "generative_model"
    training.debug_groups = 10
    training.enable_debug = True

    config.fast_fid = fast_fid = ml_collections.ConfigDict()
    fast_fid.begin_step = 10000
    fast_fid.end_step = 1000000
    fast_fid.num_samples = 1000
    fast_fid.batch_size = 64
    fast_fid.save_path = "./data/test/"
    fast_fid.ds_state_file = "E:\jikewct\Dataset\cifar10\cifar-10-images\\train_fid_stats.npz"

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.sample_steps = 500
    sampling.enable_debug = False
    sampling.log_freq = 20

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

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "mnist"
    data.img_size = (28, 28)
    data.img_channels = 1
    data.num_classes = 10
    data.root_path = "E:\jikewct\Dataset\mnist"

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = ""
    model.nn_name = ""
    model.activation = "silu"
    model.base_channels = 128
    model.time_emb_dim = 512
    model.time_emb_scale = 1.0
    model.dropout = 0.1
    model.attention_resolutions = (1, )
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
    model.embedding_type = "fourier"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()

    optim.decay_num = 2
    optim.weight_decay = 0.1
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0
    optim.amsgrad = False

    config.seed = 42
    config.device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    config.pipeline = ""
    return config
