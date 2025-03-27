import numpy as np
import pytest

from numtorch.layers import TanhActivation
from numtorch.utils.context import eval


def nice_number(x: float) -> float:
    """Returns a number that can be given to
    tanh to get a simple output"""
    return np.log(1 / np.sqrt(np.abs(x)))


def test_constructor():
    layer = TanhActivation()
    assert layer.name == "TanhActivation"
    assert layer.trainable == False
    layer.name = "Customised name"
    assert layer.name == "Customised name"


def test_input_validation():
    # Building the layer
    layer = TanhActivation()

    # Type int is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'int'> instead",
    ) as exc:
        layer(8)

    # Type float is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'float'> instead",
    ) as exc:
        layer(8.0)

    # Does not accept scalars
    with pytest.raises(
        ValueError,
        match="Expected a 1D or 2D input, received 0D instead",
    ) as exc:
        layer(np.array(8))

    # Vectors are accepted
    result = layer(np.array([1, 2]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

    # Batches of vectors are accepted
    result = layer(np.array([[1, 2], [3, 4], [5, 6]]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)

    # 3D tensors or higher are rejected
    with pytest.raises(
        ValueError,
        match="Expected a 1D or 2D input, received 3D instead",
    ):
        layer(np.array([[[1, 2]]]))


def test_forward():
    # Create Layer
    layer = TanhActivation()

    # First Test for vector
    vector_1 = np.array([nice_number(1), nice_number(2)])
    expected_1 = np.array([0, -1 / 3])
    output_1 = layer._forward(vector_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer(vector_1), expected_1)

    # Second Test for vector
    vector_2 = np.array([nice_number(3), nice_number(4)])
    expected_2 = np.array([-0.5, -0.6])
    output_2 = layer._forward(vector_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer(vector_2), expected_2)

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_output = layer._forward(batch_input)
    batch_expected = np.array([expected_1, expected_2])
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer(batch_input), batch_expected)


def test_grad_methods():
    # Create Layer
    layer = TanhActivation()

    # First Test for vector
    vector_1 = np.array([nice_number(1), nice_number(2)])
    expected_1 = np.array([[1, 0], [0, 8 / 9]])
    layer(vector_1)
    assert np.allclose(layer._grad(vector_1), expected_1)
    assert np.allclose(layer.inputs_grad, expected_1)
    assert layer.parameters_grad == {}

    # Second Test for vector
    vector_2 = np.array([nice_number(3), nice_number(4)])
    expected_2 = np.array([[3 / 4, 0], [0, 16 / 25]])
    layer(vector_2)
    assert np.allclose(layer._grad(vector_2), expected_2)
    assert np.allclose(layer.inputs_grad, expected_2)
    assert layer.parameters_grad == {}

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    assert np.allclose(layer._grad(batch_input), batch_expected)
    assert np.allclose(layer.inputs_grad, batch_expected)
    assert layer.parameters_grad == {}


def test_disabled_training():
    # Create Layer
    layer = TanhActivation()
    assert layer.trainable == False

    # Can't make the layer trainable
    layer.trainable = True
    assert layer.trainable == False


def test_eval_mode():
    # Create Layer
    layer = TanhActivation()
    assert layer.trainable == False

    # Testing vector
    vector = np.array([nice_number(1), nice_number(2)])
    expected = np.array([0, -1 / 3])

    # No grad computation
    with eval():
        output = layer(vector)
    assert np.allclose(output, expected)
    assert layer.inputs_grad is None
    assert layer.parameters_grad == {}
