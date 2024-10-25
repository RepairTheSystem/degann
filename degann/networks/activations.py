from typing import Callable

import tensorflow as tf
import torch
import torch.nn.functional as F
import tensorflow as tf

def perceptron_threshold(x, threshold: float = 1.0):
    return tf.where(x >= threshold, 1.0, 0.0)

def torch_perceptron_threshold(x, threshold: float = 1.0):
    return torch.where(x >= threshold, torch.tensor(1.0), torch.tensor(0.0))

def torch_parabolic(x: torch.Tensor, beta: float = 0, p: float = 1 / 5):
    """
    PyTorch реализация функции активации "parabolic".

    Parameters
    ----------
    x: torch.Tensor
        Input data vector
    beta: float
        Offset along the OY axis
    p: float
        Focal parabola parameter

    Returns
    -------
    new_x: torch.Tensor
        Data vector after applying activation function
    """
    return torch.where(x >= 0, beta + torch.sqrt(2 * p * x), beta - torch.sqrt(-2 * p * x))


def parabolic(x: tf.Tensor, beta: float = 0, p: float = 1 / 5):
    """
    Activation function is described in https://rairi.frccsc.ru/en/publications/426

    Parameters
    ----------
    x: tf.Tensor
        Input data vector
    beta: float
        Offset along the OY axis
    p: float
        Focal parabola parameter

    Returns
    -------
    new_x: tf.Tensor
        Data vector after applying activation function
    """
    return tf.where(x >= 0, beta + tf.sqrt(2 * p * x), beta - tf.sqrt(-2 * p * x))


_tf_activation_functions = {
    "elu": tf.keras.activations.elu,
    "relu": tf.keras.activations.relu,
    "gelu": tf.keras.activations.gelu,
    "selu": tf.keras.activations.selu,
    "exponential": tf.keras.activations.exponential,
    "linear": tf.keras.activations.linear,
    "sigmoid": tf.keras.activations.sigmoid,
    "hard_sigmoid": tf.keras.activations.hard_sigmoid,
    "swish": tf.keras.activations.swish,
    "tanh": tf.keras.activations.tanh,
    "softplus": tf.keras.activations.softplus,
    "softsign": tf.keras.activations.softsign,
    "parabolic": parabolic,
}

_torch_activation_functions = {
    "relu": F.relu,
    "elu": F.elu,
    "gelu": F.gelu,
    "selu": F.selu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "softsign": lambda x: x / (1 + torch.abs(x)),
    "parabolic": torch_parabolic,
    "perceptron_threshold": torch_perceptron_threshold,
}

class ActivationFactory:
    """
    Factory for obtaining activation functions for TensorFlow and PyTorch.

    Parameters
    ----------
    framework : strategy
        The name of the framework that requires the activation function.
        Supported values: 'tensorflow', 'pwtorch'.
    """

    def __init__(self, framework: str):
        """
        Initializing the factory specifying the framework

        Parameter
        ----------
        framework : strategy
            The name of the framework that the factory will be used for.
            The following values are supported: 'tensorflow' for Thensorflow, 'pytorch' for Pwtorch.
        """
        self.framework = framework.lower()

    def get_activation(self, name: str) -> Callable:
        """
        Getting the activation function by name.

        Parameters
        ----------
        name : string
            The name of the activation function to receive.

        Returns
        -------
        activation_function : Callable
            The activation function of the corresponding framework.

        Exceptions
        ----------
        Value Error
            If the framework is not supported or the activation function with the specified name is not found.
        """
        if self.framework == 'tensorflow':
            return _tf_activation_functions.get(name)
        elif self.framework == 'pytorch':
            return _torch_activation_functions.get(name)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def get_all_activation_functions(self) -> dict[str, Callable]:
        """
        Returns all available activation functions for the selected framework.

        returns
        -------
        activations : dict[str, Callable]
        A dictionary containing all activation functions for TensorFlow or PyTorch, depending on the framework.

        Exceptions
        ----------
        ValueError
            If the framework is not supported.
        """
        if self.framework == 'tensorflow':
            return _tf_activation_functions
        elif self.framework == 'pytorch':
            return _torch_activation_functions
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
