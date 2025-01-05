from typing import Optional, Tuple

from .nn_code import decode, default_alphabet
from degann.networks import imodel
from degann.search_algorithms.generate import random_generate
from .search_algorithms_parameters import (
    RandomEarlyStoppingSearchParameters,
    RandomSearchParameters,
)
from .utils import update_random_generator, log_search_step


def random_search(
    parameters: RandomSearchParameters,
) -> Tuple[float, int, str, str, dict]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    parameters: RandomSearchParameters
        Search algorithm parameters

    Returns
    -------
    search_results: tuple[float, int, str, str, dict]
        Results of the algorithm are described by these parameters

        best_loss: float
            The value of the loss function during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
    """
    if parameters.nn_alphabet is None:
        parameters.nn_alphabet = default_alphabet

    best_net: dict
    best_loss = 1e6
    best_epoch: int
    for i in range(parameters.iterations):
        gen = random_generate(
            min_epoch=parameters.min_epoch,
            max_epoch=parameters.max_epoch,
            min_length=parameters.nn_min_length,
            max_length=parameters.nn_max_length,
            alphabet=parameters.nn_alphabet,
            block_size=parameters.nn_alphabet_block_size,
        )

        b, a = decode(
            gen[0].value(),
            block_size=parameters.nn_alphabet_block_size,
            offset=parameters.nn_alphabet_offset,
        )
        curr_best = imodel.IModel(
            parameters.input_size, b, parameters.output_size, a + ["linear"]
        )
        curr_best.compile(
            optimizer=parameters.optimizer, loss_func=parameters.loss_function
        )
        curr_epoch = gen[1].value()
        hist = curr_best.train(
            parameters.data[0],
            parameters.data[1],
            epochs=curr_epoch,
            verbose=0,
            callbacks=parameters.callbacks,
        )
        curr_loss = hist.history["loss"][-1]
        curr_val_loss = (
            curr_best.evaluate(
                parameters.val_data[0],
                parameters.val_data[1],
                verbose=0,
                return_dict=True,
            )["loss"]
            if parameters.val_data is not None
            else None
        )

        if parameters.logging:
            fn = f"{parameters.file_name}_{len(parameters.data[0])}_0_{parameters.loss_function}_{parameters.optimizer}"
            log_search_step(
                model=curr_best,
                activations=a,
                code=gen[0].value(),
                epoch=gen[1].value(),
                optimizer=parameters.optimizer,
                loss_function=parameters.loss_function,
                loss=curr_loss,
                validation_loss=curr_val_loss,
                file_name=fn,
            )

        if curr_loss < best_loss:
            best_epoch = curr_epoch
            best_net = curr_best.to_dict()
            best_loss = curr_loss
    return (
        best_loss,
        best_epoch,
        parameters.loss_function,
        parameters.optimizer,
        best_net,
    )


def random_search_endless(
    parameters: RandomEarlyStoppingSearchParameters, verbose: bool = False
) -> Tuple[float, int, str, str, dict, int]:
    """
    Algorithm for random search in the space of parameters of neural networks

    Parameters
    ----------
    parameters: RandomEarlyStoppingSearchParameters
        Search algorithm parameters
    verbose: bool
        If True, it will show additional information when searching

    Returns
    -------
    search_results: tuple[float, int, str, str, dict, int]
        Results of the algorithm are described by these parameters

        best_loss: float
            The value of the loss function during training of the best neural network
        best_epoch: int
            Number of training epochs for the best neural network
        best_loss_func: str
            Name of the loss function of the best neural network
        best_opt: str
            Name of the optimizer of the best neural network
        best_net: dict
            Best neural network presented as a dictionary
        last_iter: int
            Count of iterations in search algorithm
    """
    if parameters.nn_alphabet is None:
        parameters.nn_alphabet = default_alphabet

    nn_loss, nn_epoch, loss_f, opt_n, net = random_search(parameters)
    i = 1
    best_net = net
    best_loss = nn_loss
    best_epoch = nn_epoch
    while nn_loss > parameters.loss_threshold and i != parameters.max_launches:
        if verbose:
            print(
                f"Random search until less than threshold. Last loss = {nn_loss}. Iterations = {i}"
            )
        nn_loss, nn_epoch, loss_f, opt_n, net = random_search(parameters)
        i += 1
        if nn_loss < best_loss:
            best_net = net
            best_loss = nn_loss
            best_epoch = nn_epoch
    return (
        best_loss,
        best_epoch,
        parameters.loss_function,
        parameters.optimizer,
        best_net,
        i,
    )
