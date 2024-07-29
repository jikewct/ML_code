from .base_dataset import BaseDataset

_DATASETS = {}


def register_dataset(cls=None, *, name=None):
    """A decorator for registering dataset classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _DATASETS:
            raise ValueError(f"Already registered dataset with name: {local_name}")
        _DATASETS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_dataset(name):
    return _DATASETS[name]


def create_dataset(config) -> BaseDataset:
    """Create  dataset."""
    dataset_name = config.data.dataset
    dataset = get_dataset(dataset_name)(**config.data)
    # score_dataset = score_dataset.to(config.device)
    # score_dataset = torch.nn.DataParallel(score_dataset)
    return dataset
