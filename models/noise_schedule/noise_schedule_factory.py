from .base_noise_schedule import BaseNoiseSchedule

_NOISE_SCHEDULERS = {}


def register_noise_scheduler(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _NOISE_SCHEDULERS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _NOISE_SCHEDULERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_noise_scheduler(name):
    return _NOISE_SCHEDULERS[name]


def create_noise_scheduler(config, *args) -> BaseNoiseSchedule:
    """Create  model."""
    model_name = config.model.name
    scheduler_name = config.model[model_name].scheduler
    scheduler_config = config.model[scheduler_name]
    if hasattr(scheduler_config, "schedule_type"):
        scheduler_name = scheduler_config["schedule_type"] + "_" + scheduler_name
    scheduler = get_noise_scheduler(scheduler_name)(*args, **scheduler_config)
    # score_model = score_model.to(config.device)
    # score_model = torch.nn.DataParallel(score_model)
    return scheduler
