import numpy as np
import pytest

from numtorch.layers import MSELoss
from numtorch.tests.helpers import create_perceptron_model
from numtorch.utils.context import eval


def test_constructor():
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)
    assert loss.model == model
    assert loss.name == "MSELoss"
    assert loss.trainable == False
    loss.name = "Customised name"
    assert loss.name == "Customised name"


def test_input_validation():
    # Create Layer
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)

    # Test wrong type
    with pytest.raises(
        TypeError,
        match="Expected inputs of type np.ndarray",
    ) as exc:
        loss(8, 5)

    # Test Valid arrays
    y = np.array([3, 2])
    target = np.array([1, 2])
    result = loss(y, target)
    assert isinstance(result, float)

    # Test Shape mismatch
    y = np.array([[3, 2], [2.5, 6]])
    target = np.array([1, 2])
    with pytest.raises(
        ValueError,
        match="Shape mismatch between preds and targets",
    ) as exc:
        loss(y, target)

    # Test two tensors
    y = np.array([[3, 2], [2.5, 6]])
    target = np.array([[4, 2], [4.5, 6]])
    result = loss(y, target)
    assert isinstance(result, float)


def test_forward():
    # Create Layer
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)

    # Test Valid arrays
    y = np.array([3, 2])
    target = np.array([1, 2])
    result = loss(y, target)
    assert result == 2.0

    # Test two tensors
    y = np.array([[3, 2], [2.5, 6]])
    target = np.array([[4, 2], [4.5, 6]])
    result = loss(y, target)
    assert result == 1.25


def test_grad_methods():
    # Create Layer
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)

    # Test Valid arrays
    y = np.array([3, 2])
    target = np.array([1, 2])
    expected = np.array([2.0, 0])
    loss(y, target)
    assert np.allclose(loss._grad(y, target), expected)
    assert np.allclose(loss.inputs_grad, expected)
    assert loss.parameters_grad == {}

    # Test two tensors
    y = np.array([[3, 2], [6, 6]])
    target = np.array([[4, 2], [5, 6]])
    expected = np.array([[-0.5, 0], [0.5, 0]])
    loss(y, target)
    assert np.allclose(loss._grad(y, target), expected)
    assert np.allclose(loss.inputs_grad, expected)
    assert loss.parameters_grad == {}


def test_backward():
    assert True


def test_disabled_training():
    # Create Layer
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)
    assert loss.trainable == False

    # Can't make the layer untrainable
    loss.trainable = True
    assert loss.trainable == False


def test_eval_mode():
    # Create Layer
    model = create_perceptron_model(2, 3, 2)
    loss = MSELoss(model)
    assert loss.trainable == False

    # Testing vector
    y_vector = np.array([-2.3, 2.5])
    pred_vector = np.array([1.7, -0.5])
    expected = 12.5

    # No grad computation
    with eval():
        output = loss(y_vector, pred_vector)

    assert output == expected
    assert loss.inputs_grad is None
    assert loss.parameters_grad == {}
