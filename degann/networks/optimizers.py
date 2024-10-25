from typing import Callable
import torch.optim as optim
from tensorflow import keras

_optimizers: dict = {
    "Adadelta": keras.optimizers.Adadelta,
    "Adafactor": keras.optimizers.Adafactor,
    "Adagrad": keras.optimizers.Adagrad,
    "Adam": keras.optimizers.Adam,
    "AdamW": keras.optimizers.AdamW,
    "Adamax": keras.optimizers.Adamax,
    "Ftrl": keras.optimizers.Ftrl,
    "Lion": keras.optimizers.Lion,
    "LossScaleOptimizer": keras.optimizers.LossScaleOptimizer,
    "Nadam": keras.optimizers.Nadam,
    "RMSprop": keras.optimizers.RMSprop,
    "SGD": keras.optimizers.SGD,
}

_torch_optimizers: dict = {
    "Adadelta": optim.Adadelta,
    "Adagrad": optim.Adagrad,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "Adamax": optim.Adamax,
    "RMSprop": optim.RMSprop,
    "SGD": optim.SGD,
}

class OptimizerFactory:
    def __init__(self, framework: str):
        self.framework = framework.lower()

    def get_optimizer(self, name: str, **kwargs):
        """
        Gets the optimizer by name.
        
        Parameters
        ----------
        name: str
            The name of the optimizer
        kwargs: dict
            Additional parameters for initializing the optimizer

        Returns
        ----------
        optimizer_class: Callable
            An instance of the optimizer
        """
        if self.framework == 'tensorflow':
            optimizer_class = _optimizers.get(name)
            if optimizer_class:
                return optimizer_class(**kwargs)
            else:
                raise ValueError(f"Optimizer {name} not found for TensorFlow.")
        elif self.framework == 'pytorch':
            optimizer_class = _torch_optimizers.get(name)
            if optimizer_class:
                return optimizer_class(**kwargs)
            else:
                raise ValueError(f"Optimizer {name} not found for PyTorch.")
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def get_all_optimizers(self):
        if self.framework == 'tensorflow':
            return _optimizers
        elif self.framework == 'pytorch':
            return _torch_optimizers
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
            # вынести из фабрики вывод из словаря
