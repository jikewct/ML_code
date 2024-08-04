from .base_sample import BaseSample

_SAMPLERS = {}


def register_sampler(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _SAMPLERS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _SAMPLERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_sampler(name):
    return _SAMPLERS[name]


def create_sampler(model, config) -> BaseSample:
    """Create  model."""
    sampling_method = config.sampling.method
    sampling_config = config.sampling[sampling_method]
    sampler = get_sampler(sampling_method)(model, sampling_method, **sampling_config)
    # score_model = score_model.to(config.device)
    # score_model = torch.nn.DataParallel(score_model)
    return sampler
