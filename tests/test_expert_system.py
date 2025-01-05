import pytest

import numpy as np

from degann.expert import ExpertSystemTags, suggest_parameters, execute_pipeline
from degann.expert.tags import EquationType, ModelPredictTime, DataSize, RequiredModelPrecision
from degann.networks import IModel


@pytest.fixture
def train_file_name():
    return "exp_150_train.csv"

@pytest.fixture
def validate_file_name():
    return "exp_150_validate.csv"

@pytest.fixture
def equation_data(train_file_name, validate_file_name):
    folder_path = "./tests/data"
    train_data = np.genfromtxt(folder_path + "/" + train_file_name, delimiter=",")
    train_data_x, train_data_y = train_data[:, 0], train_data[:, 1]
    train_data_x = train_data_x.reshape((1, -1)).T
    train_data_y = train_data_y.reshape((1, -1)).T

    validation_data = np.genfromtxt(folder_path + "/" + validate_file_name, delimiter=",")
    validation_data_x, validation_data_y = validation_data[:, 0], validation_data[:, 1]
    validation_data_x = validation_data_x.reshape((1, -1)).T
    validation_data_y = validation_data_y.reshape((1, -1)).T

    return ((train_data_x, train_data_y), (validation_data_x, validation_data_y))


def test_expert_system(equation_data):
    train_data_x = equation_data[0][0]
    train_data_y = equation_data[0][1]

    validation_data_x = equation_data[1][0]
    validation_data_y = equation_data[1][1]

    nn_1_32_16_8_1 = IModel(input_size=1, block_size=[32, 16, 8], output_size=1)
    nn_1_32_16_8_1.compile(
        optimizer="Adam",
        loss_func="MaxAbsoluteDeviation",
        metrics=[],
    )
    model_val_loss = nn_1_32_16_8_1.evaluate(validation_data_x, validation_data_y, verbose=0)

    selector_tags = ExpertSystemTags()
    selector_tags.equation_type = EquationType.EXP
    selector_tags.model_precision = RequiredModelPrecision.MINIMAL
    selector_tags.predict_time = ModelPredictTime.MEDIUM
    selector_tags.data_size = DataSize.MEDIAN
    algorithms_parameters = suggest_parameters(
        tags=selector_tags
    )
    algorithms_parameters.loss_function = "MaxAbsoluteDeviation"

    result_loss, result_nn = execute_pipeline(
        input_size=1,
        output_size=1,
        data=(train_data_x, train_data_y),
        parameters=algorithms_parameters,
    )

    model_from_expert_system = IModel(1, [], 1)
    model_from_expert_system.from_dict(result_nn)  # restore model from dict
    model_from_expert_system.compile(
        optimizer="Adam",
        loss_func="MaxAbsoluteDeviation",
        metrics=[],
    )

    expert_val_loss = model_from_expert_system.evaluate(validation_data_x, validation_data_y, verbose=0)

    assert expert_val_loss < model_val_loss
