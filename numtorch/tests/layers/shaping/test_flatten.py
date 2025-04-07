import numpy as np
import pytest

from numtorch.layers import Flatten
from numtorch.utils.context import eval


def test_constructor():
    layer = Flatten(2)
    assert layer.name == "Flatten"
    assert layer.trainable == False
    assert layer.ndim == 2
    layer.name = "Customised name"
    assert layer.name == "Customised name"


def test_input_validation():
    # Building the layer
    layer = Flatten(2)

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
        match="Expected a 2D input or a 3D batch, received 0D instead",
    ) as exc:
        layer(np.array(8))

    # Vectors are not accepted
    with pytest.raises(
        ValueError,
        match="Expected a 2D input or a 3D batch, received 1D instead",
    ) as exc:
        layer(np.array([1, 2, 3]))

    # 2D Matrices are accepted
    result = layer(np.array([[1, 2], [3, 4]]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)

    # Batches of 2D matrices are accepted
    result = layer(np.array([[[1, 2], [3, 4], [5, 6]]]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 6)

    # 4D tensors or higher are rejected
    with pytest.raises(
        ValueError,
        match="Expected a 2D input or a 3D batch, received 4D instead",
    ):
        layer(np.array([[[[1, 2]]]]))


def test_forward():
    """Flattening 2D matrices"""

    # Create Layer
    layer = Flatten(2)

    # First Test for vector
    matrix_1 = np.array([[1, 2], [3, 4]])
    expected_1 = np.array([1, 2, 3, 4])
    output_1 = layer._forward(matrix_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer(matrix_1), expected_1)

    # Second Test for vector
    matrix_2 = np.array([[0.3, -8], [4, 2.5]])
    expected_2 = np.array([0.3, -8, 4, 2.5])
    output_2 = layer._forward(matrix_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer(matrix_2), expected_2)

    # Test with batch
    batch_input = np.array([matrix_1, matrix_2])
    batch_output = layer._forward(batch_input)
    batch_expected = np.array([expected_1, expected_2])
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer(batch_input), batch_expected)


def test_forward_2():
    """Flattening 3D tensors"""

    # Create Layer
    layer = Flatten(3)

    # First Test for vector
    matrix_1 = np.array([[[1, 2], [3, 4]]])
    expected_1 = np.array([1, 2, 3, 4])
    output_1 = layer._forward(matrix_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer(matrix_1), expected_1)

    # Second Test for vector
    matrix_2 = np.array([[[0.3, -8], [4, 2.5]]])
    expected_2 = np.array([0.3, -8, 4, 2.5])
    output_2 = layer._forward(matrix_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer(matrix_2), expected_2)

    # Test with batch
    batch_input = np.array([matrix_1, matrix_2])
    batch_output = layer._forward(batch_input)
    batch_expected = np.array([expected_1, expected_2])
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer(batch_input), batch_expected)


def test_grad_methods_1():
    """Flattening 2D matrices"""

    # Create Layer
    layer = Flatten(2)

    # First Test for vector
    matrix_1 = np.array([[1, 2, 3], [4, 5, 6]])
    expected_1 = np.array(
        [
            [[1, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 0, 1]],
        ]
    )
    layer(matrix_1)
    assert np.allclose(layer._grad(matrix_1), expected_1)
    assert np.allclose(layer.inputs_grad, expected_1)
    assert layer.parameters_grad == {}

    # Second Test for vector
    matrix_2 = np.array([[0.3, -8, 2.4], [4, 2.5, -7.33]])
    expected_2 = expected_1
    layer(matrix_2)
    assert np.allclose(layer._grad(matrix_2), expected_2)
    assert np.allclose(layer.inputs_grad, expected_2)
    assert layer.parameters_grad == {}

    # Test with batch
    batch_input = np.array([matrix_1, matrix_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    assert np.allclose(layer._grad(batch_input), batch_expected)
    assert np.allclose(layer.inputs_grad, batch_expected)
    assert layer.parameters_grad == {}


def test_grad_methods_2():
    """Flattening 3D tensors"""

    # Create Layer
    layer = Flatten(3)

    # First Test for vector
    matrix_1 = np.array([[[1, 2, 3], [4, 5, 6]]])
    expected_1 = np.array(
        [
            [[[1, 0, 0], [0, 0, 0]]],
            [[[0, 1, 0], [0, 0, 0]]],
            [[[0, 0, 1], [0, 0, 0]]],
            [[[0, 0, 0], [1, 0, 0]]],
            [[[0, 0, 0], [0, 1, 0]]],
            [[[0, 0, 0], [0, 0, 1]]],
        ]
    )
    layer(matrix_1)
    assert np.allclose(layer._grad(matrix_1), expected_1)
    assert np.allclose(layer.inputs_grad, expected_1)
    assert layer.parameters_grad == {}

    # Second Test for vector
    matrix_2 = np.array([[[0.3, -8, 2.4], [4, 2.5, -7.33]]])
    expected_2 = expected_1
    layer(matrix_2)
    assert np.allclose(layer._grad(matrix_2), expected_2)
    assert np.allclose(layer.inputs_grad, expected_2)
    assert layer.parameters_grad == {}

    # Test with batch
    batch_input = np.array([matrix_1, matrix_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    assert np.allclose(layer._grad(batch_input), batch_expected)
    assert np.allclose(layer.inputs_grad, batch_expected)
    assert layer.parameters_grad == {}


def test_disabled_training():
    # Create Layer
    layer = Flatten(2)
    assert layer.trainable == False

    # Can't make the layer trainable
    layer.trainable = True
    assert layer.trainable == False


def test_eval_mode():
    # Create Layer
    layer = Flatten(2)
    assert layer.trainable == False

    # Testing vector
    matrix = np.array([[1, 2], [3, 4]])
    expected = np.array([1, 2, 3, 4])

    # No grad computation
    with eval():
        output = layer(matrix)
    assert np.allclose(output, expected)
    assert layer.inputs_grad is None
    assert layer.parameters_grad == {}
