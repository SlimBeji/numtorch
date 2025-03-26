import numpy as np

from numtorch.utils import (
    sigmoid_explicit_jacobian,
    sigmoid_jacobian,
    tanh_explicit_jacobian,
    tanh_jacobian,
)


def test_sigmoid():
    # Vectors
    for _ in range(5):
        vector = np.random.randn(4)
        assert np.allclose(sigmoid_explicit_jacobian(vector), sigmoid_jacobian(vector))

    # Batch of vectors
    for _ in range(5):
        batch = np.random.randn(4, 3)
        assert np.allclose(sigmoid_explicit_jacobian(batch), sigmoid_jacobian(batch))


def test_tanh():
    for _ in range(5):
        vector = np.random.randn(4)
        assert np.allclose(tanh_explicit_jacobian(vector), tanh_jacobian(vector))

    # Batch of vectors
    for _ in range(5):
        batch = np.random.randn(4, 3)
        assert np.allclose(tanh_explicit_jacobian(batch), tanh_jacobian(batch))
