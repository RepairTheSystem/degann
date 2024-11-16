from typing import Callable, Dict, Optional
import torch.optim as optim
from ..config import _framework  
from tensorflow.keras import optimizers as tf_optimizers

_optimizer_name: Dict[str, Callable] = {}

def _initialize_optimizer(name: str):
    """
    Initializes the optimizer and adds it to _optimizer_name.

    Parameters
    ----------
    name: string
        The name of the optimizer.
    """
    global _optimizer_name

    if _framework == 'tensorflow':
        optimizers = {
            "Adadelta": tf_optimizers.Adadelta,
            "Adafactor": tf_optimizers.experimental.Adafactor,
            "Adagrad": tf_optimizers.Adagrad,
            "Adam": tf_optimizers.Adam,
            "AdamW": tf_optimizers.AdamW,
            "Adamax": tf_optimizers.Adamax,
            "Ftrl": tf_optimizers.Ftrl,
            "Lion": tf_optimizers.experimental.Lion,
            "Nadam": tf_optimizers.Nadam,
            "RMSprop": tf_optimizers.RMSprop,
            "SGD": tf_optimizers.SGD,
        }
    elif _framework == 'pytorch':
        optimizers = {
            "Adadelta": optim.Adadelta,
            "Adagrad": optim.Adagrad,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW,
            "Adamax": optim.Adamax,
            "RMSprop": optim.RMSprop,
            "SGD": optim.SGD,
        }
    else:
        raise ValueError(f"Unsupported framework: {_framework}")

    if name in optimizers:
        _optimizer_name[name] = optimizers[name]
    else:
        raise ValueError(f"Optimizer '{name}' is not supported by {_framework}")


def get_optimizer(name: str, **kwargs) -> Optional[Callable]:
    """
    Returns the optimizer by name, automatically initializing it if necessary.

    Parameters
    ----------
    name: string
        The name of the optimizer.
    kwargs: dict
        Parameters for initializing the optimizer.

    Returns
    -------
    optimizer: Optional[Callable]
        Initialized optimizer or None if the name is not found.
    """
    if name not in _optimizer_name:
        _initialize_optimizer(name)
    optimizer_class = _optimizer_name.get(name)
    if optimizer_class:
        return optimizer_class(**kwargs)
    else:
        return None


def get_all_optimizers() -> Dict[str, Callable]:
    """
    Returns all available optimizers, automatically initializing them if necessary.

    Returns
    -------
    optimizers: Dict[str, Callable]
        A dictionary of all available optimizers.
    """
    if not _optimizer_name:
        for opt_name in ["Adadelta", "Adafactor", "Adagrad", "Adam", "AdamW", "Adamax",
                         "Ftrl", "Lion", "Nadam", "RMSprop", "SGD"]:
            _initialize_optimizer(opt_name)
    return _optimizer_name