from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from ..config import _framework  
import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn.functional as F


# Base class for loss functions
class BaseLoss(ABC):
    """
    A base class that mimics `tf.keras.losses.Loss` but supports multiple frameworks like TensorFlow and PyTorch.
    """

    def __init__(self, reduction: str = "none", name: str = "base_loss"):
        self.reduction = reduction
        self.name = name

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass

    def reduce_loss(self, loss):
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return tf.reduce_sum(loss) if isinstance(loss, tf.Tensor) else torch.sum(loss)
        elif self.reduction == "mean":
            return tf.reduce_mean(loss) if isinstance(loss, tf.Tensor) else torch.mean(loss)
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction}")

# Empty dictionaries for storing loss functions
_loss_registry: Dict[str, Callable] = {}

# TensorFlow-specific loss functions
def _initialize_tf_losses():
    return {
        "Huber": keras.losses.Huber(),
        "LogCosh": keras.losses.LogCosh(),
        "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
        "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
        "MeanSquaredError": keras.losses.MeanSquaredError(),
        "RootMeanSquaredError": RMSE(),
        "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
        "RelativeAbsoluteError": RelativeAbsoluteError(),
        "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
        "MaxAbsolutePercentageError": MaxAbsolutePercentageError(),
    }

# PyTorch-specific loss functions
def _initialize_torch_losses():
    return {
        "RelativeAbsoluteError": PyTorchRelativeAbsoluteError(),
        "MaxAbsoluteDeviation": PyTorchMaxAbsoluteDeviation(),
        "RootMeanSquaredError": PyTorchRMSE(),
    }


# Custom loss classes for TensorFlow
class RelativeAbsoluteError(BaseLoss):
    def __call__(self, y_true, y_pred):
        true_mean = tf.reduce_mean(y_true)
        numerator = tf.reduce_sum(tf.abs(y_true - y_pred))
        denominator = tf.reduce_sum(tf.abs(y_true - true_mean))
        denominator = tf.where(denominator == 0.0, tf.constant(1.0), denominator)
        loss = numerator / denominator
        return self.reduce_loss(loss)


class MaxAbsoluteDeviation(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = tf.reduce_max(tf.abs(y_true - y_pred))
        return self.reduce_loss(loss)


class MaxAbsolutePercentageError(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = tf.reduce_max(tf.abs((y_true - y_pred) / y_true)) * 100.0
        return self.reduce_loss(loss)


class RMSE(BaseLoss):
    def __call__(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_pred - y_true))
        loss = tf.sqrt(mse)
        return self.reduce_loss(loss)


# Custom loss classes for PyTorch
class PyTorchRelativeAbsoluteError(BaseLoss):
    def __call__(self, y_true, y_pred):
        true_mean = torch.mean(y_true)
        numerator = torch.sum(torch.abs(y_true - y_pred))
        denominator = torch.sum(torch.abs(y_true - true_mean))
        denominator = denominator if denominator != 0 else torch.tensor(1.0)
        loss = numerator / denominator
        return self.reduce_loss(loss)


class PyTorchMaxAbsoluteDeviation(BaseLoss):
    def __call__(self, y_true, y_pred):
        loss = torch.max(torch.abs(y_true - y_pred))
        return self.reduce_loss(loss)


class PyTorchRMSE(BaseLoss):
    def __call__(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2)
        loss = torch.sqrt(mse)
        return self.reduce_loss(loss)


# Factory to initialize and retrieve loss functions
def _initialize_loss(name: str):
    """
    Initializes the loss function based on the framework and adds it to the registry.

    Parameters
    ----------
    name : str
        The name of the loss function to initialize.
    """
    global _loss_registry

    if _framework == "tensorflow":
        tf_losses = _initialize_tf_losses()
        if name in tf_losses:
            _loss_registry[name] = tf_losses[name]
        else:
            raise ValueError(f"Loss function '{name}' is not available in TensorFlow.")
    elif _framework == "pytorch":
        torch_losses = _initialize_torch_losses()
        if name in torch_losses:
            _loss_registry[name] = torch_losses[name]
        else:
            raise ValueError(f"Loss function '{name}' is not available in PyTorch.")
    else:
        raise ValueError(f"Unsupported framework: {_framework}")


def get_loss(name: str) -> Optional[Callable]:
    """
    Retrieves a loss function by name, initializing it if necessary.

    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
    Callable
        The corresponding loss function or None if not found.
    """
    if name not in _loss_registry:
        _initialize_loss(name)
    return _loss_registry.get(name)


def get_all_losses() -> Dict[str, Callable]:
    """
    Retrieves all available loss functions, initializing them if necessary.

    Returns
    -------
    Dict[str, Callable]
        A dictionary containing all available loss functions.
    """
    if not _loss_registry:
        if _framework == "tensorflow":
            for loss_name in _initialize_tf_losses().keys():
                _initialize_loss(loss_name)
        elif _framework == "pytorch":
            for loss_name in _initialize_torch_losses().keys():
                _initialize_loss(loss_name)
    return _loss_registry
