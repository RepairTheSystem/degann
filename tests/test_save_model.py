import pytest
import numpy as np

from degann.networks.imodel import IModel
from tests.utils import array_compare, file_compare


@pytest.fixture
def folder_path():
    return "./tests/data"


@pytest.mark.parametrize(
    "inp, shape, act_init, decorator_params",
    [
        (np.array([[1]], dtype=float), [1, [1], 1], ["sigmoid", "linear"], None),
        (np.array([[1]], dtype=float), [1, [1], 1], "sigmoid", None),
        (np.array([[1]], dtype=float), [1, [1], 1], ["linear", "linear"], None),
        (np.array([[1, 1]], dtype=float), [2, [1], 1], "tanh", None),
        (np.array([[1], [1]], dtype=float), [1, [1], 1], "tanh", None),
    ],
)
def test_predict_is_same(inp, shape, act_init, decorator_params, folder_path):
    nn = IModel(
        shape[0],
        shape[1],
        shape[2],
        activation_func=act_init,
        decorator_params=decorator_params,
    )

    expected = nn.feedforward(inp).numpy()
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")
    actual = nn_loaded.feedforward(inp).numpy()

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "inp, shape",
    [
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, [1], 1],
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 2],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 1],
        ),
    ],
)
def test_file_is_same(inp, shape, folder_path):
    nn = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn.export_to_file(f"{folder_path}/test_export")

    nn_loaded = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn_loaded.from_file(f"{folder_path}/test_export")
    nn_loaded.export_to_file(f"{folder_path}/test_export1")

    assert file_compare(
        f"{folder_path}/test_export.apg", f"{folder_path}/test_export1.apg"
    )
