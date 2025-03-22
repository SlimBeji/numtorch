import numpy as np
import pytest

from numtorch.layers import ReluActivation
from numtorch.utils.context import eval


def test_constructor():
    layer = ReluActivation()
    assert layer.name == "ReluActivation"
    assert layer.trainable == False
    layer.name = "Customised name"
    assert layer.name == "Customised name"


def test_input_validation():
    # Type int is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'int'> instead",
    ) as exc:
        ReluActivation()(8)

    # Type float is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'float'> instead",
    ) as exc:
        ReluActivation()(8.0)


def test_forward():
    # Create Layer
    layer = ReluActivation()

    # Accept scalars
    scalar = np.array(8)
    expected = np.array(8)
    result = layer(scalar)
    assert np.allclose(expected, result)

    # Vectors are accepted
    vector = np.array([8, -2])
    expected = np.array([8, 0])
    result = layer(vector)
    assert np.allclose(expected, result)

    # Batches of vectors are accepted
    batch = np.array([[1, -2, 0], [-9, -1, 2]])
    expected = np.array([[1, 0, 0], [0, 0, 2]])
    result = layer(batch)
    assert np.allclose(expected, result)

    # 3D tensors or higher are also accepted
    tensor = np.array([[[1, -1]]])
    expected = np.array([[[1, 0]]])
    result = layer(tensor)
    assert np.allclose(expected, result)


def test_grad_methods():
    # Create Layer
    layer = ReluActivation()

    # Test scalar
    scalar = np.array(8)
    expected = np.array(1)
    layer(scalar)
    assert np.allclose(layer._grad(scalar), expected)
    assert np.allclose(layer.inputs_grad, expected)
    assert layer.parameters_grad == {}
    # Second test with negative value
    scalar = np.array(-8)
    expected = np.array(0)
    layer(scalar)
    assert np.allclose(layer._grad(scalar), expected)
    assert np.allclose(layer.inputs_grad, expected)
    assert layer.parameters_grad == {}

    # Test for vectors
    vector = np.array([8, -2])
    expected = np.array([1, 0])
    layer(vector)
    assert np.allclose(layer._grad(vector), expected)
    assert np.allclose(layer.inputs_grad, expected)
    assert layer.parameters_grad == {}

    # Batches of vectors are accepted
    batch = np.array([[1, -2, 0], [-9, -1, 2]])
    expected = np.array([[1, 0, 0], [0, 0, 1]])
    layer(batch)
    assert np.allclose(layer._grad(batch), expected)
    assert np.allclose(layer.inputs_grad, expected)
    assert layer.parameters_grad == {}

    # 3D tensors or higher are also accepted
    tensor = np.array([[[1, -8]]])
    expected = np.array([[[1, 0]]])
    layer(tensor)
    assert np.allclose(layer._grad(tensor), expected)
    assert np.allclose(layer.inputs_grad, expected)
    assert layer.parameters_grad == {}


def test_disabled_training():
    # Create Layer
    layer = ReluActivation()
    assert layer.trainable == False

    # Can't make the layer trainable
    layer.trainable = True
    assert layer.trainable == False


def test_eval_mode():
    # Create Layer
    layer = ReluActivation()
    assert layer.trainable == False

    # Testing vector
    vector = np.array([3.0, -2.0])
    expected_output = np.array([3, 0])

    # No grad computation
    with eval():
        output = layer(vector)
    assert np.allclose(output, expected_output)
    assert layer.inputs_grad is None
    assert layer.parameters_grad == {}
