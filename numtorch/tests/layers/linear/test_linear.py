import re

import numpy as np
import pytest

from numtorch.layers import LinearLayer
from numtorch.utils.context import eval


def test_constructor():
    layer = LinearLayer(2, 3)
    assert layer.in_feature == 2
    assert layer.out_feature == 3
    assert layer.weights.shape == (3, 2)
    assert layer.bias.shape == (3,)
    assert layer.name == "LinearLayer"
    assert layer.trainable == True
    layer.name = "Customised name"
    assert layer.name == "Customised name"


def test_input_validation():
    # Type int is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'int'> instead",
    ) as exc:
        LinearLayer(2, 3)(8)

    # Type float is not accepted
    with pytest.raises(
        TypeError,
        match="Expected input of type np.ndarray, received <class 'float'> instead",
    ) as exc:
        LinearLayer(2, 3)(8.0)

    # Does not accept scalars
    with pytest.raises(
        ValueError,
        match="Linear Layers does not accept scalars, received 8",
    ) as exc:
        LinearLayer(2, 3)(np.array(8))

    # Vectors are accepted
    result = LinearLayer(2, 3)(np.array([1, 2]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

    # Batches of vectors are accepted
    result = LinearLayer(2, 3)(np.array([[1, 2], [3, 4]]))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)

    # 3D tensors or higher are rejected
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Linear Layers does not accept tensors of dimension bigger than 2, received shape"
        ),
    ):
        LinearLayer(2, 3)(np.array([[[1, 2]]]))

    # Catch shape mismatch for vectors
    with pytest.raises(
        ValueError, match="Size mismatch: Linear Layer in_features: 3 different from 2"
    ):
        LinearLayer(3, 2)(np.array([1, 2]))

    # Catch shape mismatch for batches
    with pytest.raises(
        ValueError, match="Size mismatch: Linear Layer in_features: 3 different from 2"
    ):
        LinearLayer(3, 2)(np.array([[1, 2], [3, 4]]))


def test_forward():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])

    # First Test for vector
    vector_1 = np.array([1.0, 2.0])
    expected_1 = np.array([5.1, 11.2, 17.3])
    output_1 = layer._forward(vector_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer(vector_1), expected_1)

    # Second Test for vector
    vector_2 = np.array([3.0, 4.0])
    expected_2 = np.array([11.1, 25.2, 39.3])
    output_2 = layer._forward(vector_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer(vector_2), expected_2)

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_output = layer._forward(batch_input)
    batch_expected = np.array([expected_1, expected_2])
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer(batch_input), batch_expected)


def test_grad():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])

    # First Test for vector
    vector_1 = np.array([1.0, 2.0])
    expected_1 = layer.weights
    layer(vector_1)
    assert np.allclose(layer._grad(vector_1), expected_1)
    assert np.allclose(layer.inputs_grad, expected_1)

    # Second Test for vector
    vector_2 = np.array([3.0, 4.0])
    expected_2 = layer.weights
    layer(vector_2)
    assert np.allclose(layer._grad(vector_2), expected_2)
    assert np.allclose(layer.inputs_grad, expected_2)

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    assert np.allclose(layer._grad(batch_input), batch_expected)
    assert np.allclose(layer.inputs_grad, batch_expected)


def test_weights_grad():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])

    # First Test for vector
    vector_1 = np.array([1.0, 2.0])
    expected_1 = np.array(
        [
            [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]],
        ]
    )
    layer(vector_1)
    output_1 = layer._weights_grad(vector_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer.parameters_grad["weights"], expected_1)

    # Second Test for vector
    vector_2 = np.array([3.0, 4.0])
    expected_2 = np.array(
        [
            [[3.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]],
        ]
    )
    layer(vector_2)
    output_2 = layer._weights_grad(vector_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer.parameters_grad["weights"], expected_2)

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    batch_output = layer._weights_grad(batch_input)
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer.parameters_grad["weights"], batch_expected)


def test_bias_grad():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])

    # First Test for vector
    vector_1 = np.array([1.0, 2.0])
    expected_1 = np.eye(3)
    layer(vector_1)
    output_1 = layer._bias_grad(vector_1)
    assert np.allclose(output_1, expected_1)
    assert np.allclose(layer.parameters_grad["bias"], expected_1)

    # Second Test for vector
    vector_2 = np.array([3.0, 4.0])
    expected_2 = np.eye(3)
    layer(vector_2)
    output_2 = layer._bias_grad(vector_2)
    assert np.allclose(output_2, expected_2)
    assert np.allclose(layer.parameters_grad["bias"], expected_2)

    # Test with batch
    batch_input = np.array([vector_1, vector_2])
    batch_expected = np.array([expected_1, expected_2])
    layer(batch_input)
    batch_output = layer._bias_grad(batch_input)
    assert np.allclose(batch_output, batch_expected)
    assert np.allclose(layer.parameters_grad["bias"], batch_expected)


def test_disabled_training():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])
    layer.trainable = False
    assert layer.trainable == False

    # Testing Vector
    vector = np.array([1.0, 2.0])
    expected_output = np.array([5.1, 11.2, 17.3])
    expected_grad = layer.weights
    output = layer(vector)

    # Forward pass and input grad are computed
    assert np.allclose(output, expected_output)
    assert np.allclose(layer.inputs_grad, expected_grad)

    # Parameters grad computations skipped
    assert layer.parameters_grad == {}


def test_eval_mode():
    # Create Layer
    layer = LinearLayer(in_feature=2, out_feature=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    layer.bias = np.array([0.1, 0.2, 0.3])
    assert layer.trainable

    # Testing vector
    vector = np.array([1.0, 2.0])
    expected_output = np.array([5.1, 11.2, 17.3])

    # No grad computation
    with eval():
        output = layer(vector)
    assert np.allclose(output, expected_output)
    assert layer.inputs_grad is None
    assert layer.parameters_grad == {}

    # No grad computation also when not trainable
    layer.trainable = False
    assert layer.trainable == False
    with eval():
        output = layer(vector)
    assert np.allclose(output, expected_output)
    assert layer.inputs_grad is None
    assert layer.parameters_grad == {}
