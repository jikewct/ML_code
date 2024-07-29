import logging
from functools import partial

from torch import optim


def get_optimizer(params, optimizer, **kwargs):
    if optimizer.lower() == "adam":
        return optim.Adam(params, **kwargs[optimizer])
    elif optimizer.lower() == "adamw":
        # logging.info("adamw optimizer")
        return optim.AdamW(params, **kwargs[optimizer])
    elif optimizer.lower() == "rmsprop":
        return optim.RMSprop(params, **kwargs[optimizer])
    elif optimizer.lower() == "sgd":
        return optim.SGD(params, **kwargs[optimizer])
    else:
        raise NotImplementedError(optimizer)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR

    def fn(step):
        scale = 1
        if warmup_steps > 0:
            scale = min(step / warmup_steps, 1)
        # logging.info(f"current scale:{scale}, step:{step}")
        return scale

    # logging.info("customized lr scheduler")
    return LambdaLR(optimizer, partial(fn))


def get_lr_scheduler(optimizer, name, **kwargs):
    if name.lower() == "customized":
        return customized_lr_scheduler(optimizer, **kwargs[name])
    elif name.lower() == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(optimizer, **kwargs[name])
    else:
        raise NotImplementedError(name)
