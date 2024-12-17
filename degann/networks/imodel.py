import json
from collections import defaultdict
from typing import List, Optional, Dict, Union, Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim


from degann.networks.config_format import HEADER_OF_APG_FILE
from degann.networks.topology.tf_densenet import TensorflowDenseNet
from degann.networks.topology.tf_densenet import PtDenseNet

def _get_act_and_init(
    kwargs: dict,
    default_act,
    default_dec: Optional[List[Optional[Dict[str, float]]]],
    default_init,
):
    if kwargs.get("activation") is None:
        activation = default_act
    else:
        activation = kwargs["activation"]
        kwargs.pop("activation")

    if kwargs.get("decorator_params") is None:
        decorator_params = default_dec
    else:
        decorator_params = kwargs["decorator_params"]
        kwargs.pop("decorator_params")

    if kwargs.get("weight") is None:
        weight = default_init
    else:
        weight = kwargs["weight"]
        kwargs.pop("weight")

    if kwargs.get("biases") is None:
        biases = default_init
    else:
        biases = kwargs["biases"]
        kwargs.pop("biases")

    return activation, decorator_params, weight, biases, kwargs


class IModel(object):
    """
    Interface class for working with neural topology
    """

    def __init__(
        self,
        input_size: int,
        block_size: List[int],
        output_size: int,
        activation_func="sigmoid",
        weight_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        bias_init=tf.random_uniform_initializer(minval=-1, maxval=1),
        name="net",
        net_type="PtDendeNet",
        is_debug=False,
        **kwargs,
    ):
        self.network = _create_functions[net_type](
            input_size,
            block_size,
            activation_func=activation_func,
            weight=weight_init,
            biases=bias_init,
            output_size=output_size,
            is_debug=is_debug,
            **kwargs,
        )
        self._input_size = input_size
        self._output_size = output_size
        self._shape = block_size
        self._name = name
        self._is_debug = is_debug
        self.set_name(name)

    def compile(
        self,
        rate=1e-2,
        optimizer="SGD",
        loss_func="MeanSquaredError",
        metrics=None,
        run_eagerly=False,
    ) -> None:
        """
        Configures the model for training

        Parameters
        ----------
        rate: float
            learning rate for optimizer
        optimizer: str
            name of optimizer
        loss_func: str
            name of loss function
        metrics: list[str]
            list with metric function names
        run_eagerly: bool

        Returns
        -------

        """
        if metrics is None:
            metrics = [
                "MeanSquaredError",
                "MeanAbsoluteError",
                "MeanSquaredLogarithmicError",
            ]

        self.network.custom_compile(
            optimizer=optimizer,
            rate=rate,
            loss_func=loss_func,
            metric_funcs=metrics,
            run_eagerly=run_eagerly,
        )

    def feedforward(self, inputs: np.ndarray) -> tf.Tensor:
        """
        Return network answer for passed input by network __call__()

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector

        Returns
        -------
        outputs: tf.Tensor
            Network answer
        """

        return self.network(inputs, training=False)

    def predict(self, inputs: np.ndarray, callbacks: List = None) -> np.ndarray:
        """
        Return network answer for passed input by network predict()

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector
        callbacks: list
            List of tensorflow callbacks for predict function

        Returns
        -------
        outputs: np.ndarray
            Network answer
        """

        return self.network.predict(inputs, verbose=0, callbacks=callbacks)

    def train(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        validation_split=0.0,
        validation_data=None,
        epochs=10,
        mini_batch_size=None,
        callbacks: List = None,
        verbose="auto",
    ) -> keras.callbacks.History:
        """
        Train network on passed dataset and return training history

        Parameters
        ----------
        x_data: np.ndarray
            Array of input vectors
        y_data: np.ndarray
            Array of output vectors
        validation_split: float
            Percentage of data to validate
        validation_data: tuple[np.ndarray, np.ndarray]
            Validation dataset
        epochs: int
            Count of epochs for training
        mini_batch_size: int
            Size of batches
        callbacks: list
            List of tensorflow callbacks for fit function
        verbose: int
            Output accompanying training

        Returns
        -------
        history: tf.keras.callbacks.History
            History of training
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        temp = self.network.fit(
            x_data,
            y_data,
            batch_size=mini_batch_size,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            epochs=epochs,
            verbose=verbose,
        )
        return temp

    def evaluate(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size=None,
        callbacks: List = None,
        verbose="auto",
        **kwargs,
    ) -> Union[float, List[float]]:
        """
        Evaluate network on passed dataset and return evaluate history

        Parameters
        ----------
        x_data: np.ndarray
            Array of input vectors
        y_data: np.ndarray
            Array of output vectors
        batch_size: int
            Size of batches
        callbacks: list
            List of tensorflow callbacks for evaluate function
        verbose: int
            Output accompanying evaualing

        Returns
        -------
        history: Union[float, List[float]]
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics).
        """
        if self._is_debug:
            if callbacks is not None:
                callbacks.append(
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                )
            else:
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        f"log_{self.get_name}.csv", separator=",", append=False
                    )
                ]
        self._evaluate_history = self.network.evaluate(
            x_data,
            y_data,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs,
        )
        return self._evaluate_history

    def clear_history(self):
        del self.network.history
        del self._evaluate_history

    def export_to_cpp(
        self,
        path: str,
        array_type: str = "[]",
        path_to_compiler: str = None,
        vectorized_level: str = "none",
        **kwargs,
    ) -> None:
        """
        Export neural network as feedforward function on c++

        Parameters
        ----------
        path: str
            path to file with name, without extension
        array_type: str
            c-style or cpp-style ("[]" or "vector")
        path_to_compiler: str
            path to c/c++ compiler, if `None` then the resulting code will not be compiled
        vectorized_level: str
            Level of code vectorization
            Available levels: none, auto (the program will choose the latest level by itself),
            sse, avx, avx512f
        kwargs

        Returns
        -------

        """
        self.network.export_to_cpp(
            path,
            array_type,
            path_to_compiler,
            vectorized_level=vectorized_level,
            **kwargs,
        )

    def to_dict(self, **kwargs):
        """
        Export neural network to dictionary

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        return self.network.to_dict(**kwargs)

    def export_to_file(self, path, **kwargs):
        """
        Export neural network as parameters to file

        Parameters
        ----------
        path:
            path to file with name, without extension
        kwargs

        Returns
        -------

        """
        config = self.to_dict(**kwargs)
        with open(path + ".apg", "w") as f:
            f.write(HEADER_OF_APG_FILE + json.dumps(config, indent=2))

    def from_dict(self, config: dict, **kwargs):
        """
        Import neural network from dictionary

        Parameters
        ----------
        config: dict
            Network configuration

        Returns
        -------

        """
        self._shape = config["block_size"]
        self.network.from_dict(config, **kwargs)

    def from_file(self, path: str, **kwargs):
        """
        Import neural network as parameters from file

        Parameters
        ----------
        path:
            path to file with name, without extension
        kwargs

        Returns
        -------

        """
        with open(path + ".apg", "r") as f:
            for header in range(HEADER_OF_APG_FILE.count("\n")):
                _ = f.readline()
            config_str = ""
            for line in f:
                config_str += line
            config = json.loads(config_str)
            self.network.from_dict(config)
            self.set_name(config["name"])

    def set_name(self, name: str) -> None:
        """
        Set network name

        Parameters
        ----------
        name: str
            New name
        Returns
        -------
        None
        """
        self.network.set_name(name)
        self._name = name

    @property
    def get_name(self) -> str:
        return self._name

    @property
    def get_shape(self) -> List[int]:
        """
        Get shape for current network

        Returns
        -------
        shape: List[int]
            Network shape
        """

        return self._shape

    @property
    def get_input_size(self) -> int:
        """
        Get input vector size for current network

        Returns
        -------
        size: int
            Input vector size
        """

        return self._input_size

    @property
    def get_output_size(self) -> int:
        """
        Get output vector size for current network

        Returns
        -------
        size: int
            Output vector size
        """

        return self._output_size

    @property
    def get_activations(self) -> list:
        """
        Get list of activations for each layer

        Returns
        -------
        activations: list
        """

        return self.network.get_activations

    def __str__(self) -> str:
        """
        Get a string representation of the neural network

        Returns
        -------
        result: str
        """

        return str(self.network)

    @classmethod
    def create_neuron(
        cls, input_size: int, output_size: int, shape: list[int], **kwargs
    ):
        """
        Create neural network with passed size and sigmoid activation

        Parameters
        ----------
        input_size: int
        output_size: int
        shape: list[int]
            Sizes of hidden layers
        kwargs

        Returns
        -------
        net: imodel.IModel
            Neural network
        """
        activation, decorator_params, weight, biases, kwargs = _get_act_and_init(
            kwargs,
            "sigmoid",
            None,
            tf.random_normal_initializer(),
        )

        res = cls(
            input_size=input_size,
            block_size=shape,
            output_size=output_size,
            activation_func=activation,
            bias_init=biases,
            weight_init=weight,
            decorator_params=decorator_params,
            **kwargs,
        )

        return res


_create_functions = defaultdict(lambda: TensorflowDenseNet)
_create_functions["PtDenseNet"] = PtDenseNet


class PtIModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        block_size: List[int],
        output_size: int,
        activation_func: Callable = nn.Sigmoid,
        weight_init: Callable = nn.init.uniform_,
        bias_init: Callable = nn.init.uniform_,
        name: str = "net",
        net_type: str = "PtDenseNet",
        is_debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.network = _create_functions[net_type](
            input_size,
            block_size,
            output_size,
            activation_func=activation_func,
            weight_init=weight_init,
            bias_init=bias_init,
            **kwargs,
        )
        self._input_size = input_size
        self._output_size = output_size
        self._shape = block_size
        self._name = name
        self._is_debug = is_debug

    def compile(
        self,
        learning_rate: float = 1e-2,
        optimizer: str = "SGD",
        loss_func: str = "MSELoss",
        metrics: Optional[List[str]] = None,
    ) -> None:
        if not hasattr(nn, loss_func):
            raise ValueError(f"Loss function {loss_func} is not defined in torch.nn.")
        if not hasattr(optim, optimizer):
            raise ValueError(f"Optimizer {optimizer} is not defined in torch.optim.")

        self.loss_func = getattr(nn, loss_func)()
        self.optimizer = getattr(optim, optimizer)(self.parameters(), lr=learning_rate)
        self.metrics = metrics or []
    
    def train(
        self,
        x_data: torch.Tensor,
        y_data: Optional[torch.Tensor] = None,
        validation_split: float = 0.0,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        epochs: int = 10,
        mini_batch_size: int = None,
        callbacks: Optional[List] = None,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:

        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        if validation_split > 0.0:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        elif validation_data is not None:
            train_dataset = dataset
            val_dataset = torch.utils.data.TensorDataset(*validation_data)
        else:
            train_dataset = dataset
            val_dataset = None

        batch_size = mini_batch_size if mini_batch_size is not None else len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        if val_dataset is not None:
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        else:
            val_loader = None

        history = []
        for epoch in range(epochs):
            # Обучение
            self.network.train()
            epoch_loss = 0.0
            metric_results = defaultdict(float)

            for inputs, targets in train_loader:
                step_results = self.network.train_step(
                    (inputs, targets),
                    loss_func=self.loss_func,
                    optimizer=self.optimizer,
                    metrics=self.metrics,
                )
                epoch_loss += step_results["loss"]
                for k, v in step_results.items():
                    if k != "loss":
                        metric_results[k] += v

            num_batches = len(train_loader)
            epoch_metrics = {k: v / num_batches for k, v in metric_results.items()}
            train_loss = epoch_loss / num_batches

            # Валидация
            if val_loader is not None:
                self.network.eval()
                val_loss = 0.0
                val_metric_results = defaultdict(float)
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = self.network(inputs)
                        loss = self.loss_func(outputs, targets)
                        val_loss += loss.item()
                        if self.metrics:
                            for metric in self.metrics:
                                val_metric_results[metric.__name__] += metric(outputs, targets).item()

                num_val_batches = len(val_loader)
                val_metrics = {k: v / num_val_batches for k, v in val_metric_results.items()}
                val_loss /= num_val_batches
            else:
                val_loss = None
                val_metrics = {}

            # Запись истории
            epoch_history = {'loss': train_loss, **epoch_metrics}
            if val_loss is not None:
                epoch_history['val_loss'] = val_loss
                epoch_history.update({f'val_{k}': v for k, v in val_metrics.items()})
            history.append(epoch_history)

            if verbose:
                log = f"Эпоха {epoch+1}/{epochs}, Потеря: {train_loss:.4f}"
                if val_loss is not None:
                    log += f", Вал. потеря: {val_loss:.4f}"
                for k, v in epoch_metrics.items():
                    log += f", {k}: {v:.4f}"
                for k, v in val_metrics.items():
                    log += f", вал_{k}: {v:.4f}"
                print(log)

        return history

    def evaluate(self, x_data: torch.Tensor, y_data: torch.Tensor) -> Dict[str, float]:
        self.network.eval()
        with torch.no_grad():
            predictions = self.network(x_data)
            loss = self.loss_func(predictions, y_data).item()
            results = {"loss": loss}
            if self.metrics:
                for metric in self.metrics:
                    if callable(metric):
                        results[metric.__name__] = metric(y_data, predictions)
                    else:
                        metric_func = get_metric(metric)
                        results[metric] = metric_func(y_data, predictions)
            return results