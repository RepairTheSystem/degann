from abc import ABC, abstractmethod
from typing import Callable

import tensorflow as tf
from tensorflow import keras

import torch
import torch.nn.functional as F

def sign(x):
    return tf.where(x < 0.0, -1.0, 1.0)

class BaseLoss(ABC):
    """
    A base class created to parody tf.karas.loss.Loss,
    but with the ability to expand to multiple frameworks
    """
    def __init__(self, reduction: str = "none", name: str = "base_loss"):
        """
        The base class for implementing the loss function for TensorFlow and Pwtorch.
        
        Parameters:
        -----------
        reduction: str
            The reduction method (none, sum, mean).
        name: str
            The name of the loss function.
        """
        self.reduction = reduction
        self.name = name

    def __call__(self, y_true, y_pred):
        """
        Calculation of the loss function.
        """
        raise NotImplementedError("The __call__ method must be implemented in a subclass.")
    
    def reduce_loss(self, loss):
        """
        Applies reduction to loss.
        """
        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return tf.reduce_sum(loss) if isinstance(loss, tf.Tensor) else torch.sum(loss)
        elif self.reduction == "mean":
            return tf.reduce_mean(loss) if isinstance(loss, tf.Tensor) else torch.mean(loss)
        else:
            raise ValueError(f"Unknown type of reduction: {self.reduction}")


class RelativeAbsoluteError(BaseLoss):
    """
    This class provides RAE loss function:
    $$ RAE = \frac{\Sum^n_{i=1} |y_i - \hat(y)_i|}{\Sum^n_{i=1} |y_i - \bar(y)|}
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="rae", **kwargs):
        super(RelativeAbsoluteError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        true_mean = tf.reduce_mean(y_true)
        squared_error_num = tf.reduce_sum(tf.abs(y_true - y_pred))
        squared_error_den = tf.reduce_sum(tf.abs(y_true - true_mean))

        squared_error_den = tf.cond(
            pred=tf.equal(squared_error_den, tf.constant(0.0)),
            true_fn=lambda: tf.constant(1.0),
            false_fn=lambda: squared_error_den,
        )

        loss = squared_error_num / squared_error_den
        return loss


class MaxAbsoluteDeviation(BaseLoss):
    """
    This class provides Max Absolute Deviation loss function:
    $$ MAD = \max |y - \hat(y)|
    """

    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="my_mae", **kwargs
    ):
        super(MaxAbsoluteDeviation, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss


class MaxAbsolutePercentageError(BaseLoss):
    """
    This class provides Max Absolute Percentage Error loss function:
    $$ MAD = \max |\frac{y - \hat(y)}{y}|
    """

    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="maxAPE", **kwargs
    ):
        super(MaxAbsolutePercentageError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs((y_true - y_pred) / y_true)) * 100.0
        return loss


class RMSE(BaseLoss):
    """
    This class provides Root Mean squared Error loss function:
    $$ MAD = \sqrt{MSE}
    """

    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="RMSE", **kwargs):
        super(RMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.sqrt(tf.math.reduce_mean((y_pred - y_true) ** 2))
        return loss


class PyTorchRelativeAbsoluteError(BaseLoss):
    """
    This class provides RAE loss function:
    $$ RAE = \frac{\Sum^n_{i=1} |y_i - \hat(y)_i|}{\Sum^n_{i=1} |y_i - \bar(y)|}
    """
    def __call__(self, y_true, y_pred):
        true_mean = torch.mean(y_true)
        squared_error_num = torch.sum(torch.abs(y_true - y_pred))
        squared_error_den = torch.sum(torch.abs(y_true - true_mean))

        if squared_error_den == 0:
            squared_error_den = torch.tensor(1.0)
        
        loss = squared_error_num / squared_error_den
        return loss

class PyTorchMaxAbsoluteDeviation(BaseLoss):
    """
    This class provides MAD loss function:
    $$ MAD = max|y - \hat{y}|
    """
    def __init__(self, reduction: str = "none", name: str = "max_absolute_deviation", **kwargs):
        super(PyTorchMaxAbsoluteDeviation, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred):
        loss = torch.max(torch.abs(y_true - y_pred))
        return self.reduce_loss(loss)

class PyTorchRMSE(BaseLoss):
    """
    This class provides Root Mean squared Error loss function:
    $$ MAD = \sqrt{MSE}
    """
    def __init__(self, reduction: str = "none", name: str = "rmse", **kwargs):
        super(PyTorchRMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2)
        rmse = torch.sqrt(mse)
        return self.reduce_loss(rmse)

# Reduction should be set to None?
# Create losses dict for every framework 
_tf_losses: dict = {
    "Huber": keras.losses.Huber(),
    "LogCosh": keras.losses.LogCosh(),
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MaxAbsolutePercentageError": MaxAbsolutePercentageError(),
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "RootMeanSquaredError": RMSE(),
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
    "RelativeAbsoluteError": RelativeAbsoluteError(),
    "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
}

_torch_losses = {
    "RelativeAbsoluteError": PyTorchRelativeAbsoluteError(),
    "MaxAbsoluteDeviation": PyTorchMaxAbsoluteDeviation(),
    "RootMeanSquaredError" : PyTorchRMSE(),
}

class LossFactory:
    """
    A factory for obtaining loss functions for TensorFlow and PyTorch.

    Parameters
    ----------
    framework : str
        The name of the framework that requires the loss function. 
        Supported values: 'tensorflow', 'pytorch'.
    """
    
    def __init__(self, framework: str):
        self.framework = framework.lower()

    def get_loss(self, name: str):
        if self.framework == 'tensorflow':
            return _tf_losses.get(name)
        elif self.framework == 'pytorch':
            return _torch_losses.get(name)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def get_all_loss_functions(self):
        if self.framework == 'tensorflow':
            return _tf_losses
        elif self.framework == 'pytorch':
            return _torch_losses
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

# конфиг пакета со строковым значением
# фукнции активации -> слои -> нейронка