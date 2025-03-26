from typing import Callable

import numpy as np


def explicit_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a method that computes the Jacobian
    for a given derivative method. This method is for
    explanation purpose only as it does not leverage
    numpy broadcasting and use plain python loops"""

    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 0:
            return f(x)
        elif x.ndim == 1:
            return np.eye(len(x)) * f(x)
        elif x.ndim > 2:
            raise ValueError(
                f"Invalid input shape for the jacobian of an element wise function:\n {x}"
            )

        # x is a batch (ndim = 2)
        derivatives = f(x)
        batch_size = x.shape[0]
        input_size = x.shape[1]
        jacobian_shape = (batch_size, input_size, input_size)
        jacobian = np.zeros(jacobian_shape)
        for batch_index in range(batch_size):
            for vector_index in range(input_size):
                value = derivatives[batch_index, vector_index]
                jacobian[batch_index, vector_index, vector_index] = value

        return jacobian

    return wrapped


def jacobian(
    f: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Returns a method that computes the Jacobian
    for a given derivative method. This method uses
    numpy optimization for arithmetics"""

    def wrapped(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            # Optimization using numpy broadcasting
            eye_batch = np.expand_dims(np.eye(x.shape[1]), axis=0)
            derivatives = np.expand_dims(f(x), axis=-1)
            return eye_batch * derivatives
        elif x.ndim == 1:
            return np.eye(len(x)) * f(x)
        else:
            raise ValueError(
                f"Invalid input shape for the jacobian of an element wise function:\n {x}"
            )

    return wrapped


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_explicit_jacobian(x: np.ndarray) -> np.ndarray:
    return explicit_jacobian(sigmoid_prime)(x)


def sigmoid_jacobian(x: np.ndarray) -> np.ndarray:
    return jacobian(sigmoid_prime)(x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def tanh_explicit_jacobian(x: np.ndarray) -> np.ndarray:
    return explicit_jacobian(tanh_prime)(x)


def tanh_jacobian(x: np.ndarray) -> np.ndarray:
    return jacobian(tanh_prime)(x)


def softmax(x: np.ndarray) -> np.ndarray:
    maximums = x.max(axis=-1, keepdims=True)
    shifted = x - maximums
    exp: np.ndarray = np.exp(shifted)
    exp_sums = exp.sum(axis=-1, keepdims=True)
    return exp / exp_sums


def softmax_explicit_jacobian(x: np.ndarray) -> np.ndarray:
    """a verbose and explicit jacobian implementation"""
    if x.ndim not in [1, 2]:
        raise ValueError(
            f"Invalid input shape for the jacobian of an element wise function:\n {x}"
        )

    if x.ndim == 1:
        batch = np.expand_dims(x, axis=0)
    else:
        batch = x

    batch_size = batch.shape[0]
    input_size = batch.shape[1]
    jacobian_shape = (batch_size, input_size, input_size)
    jacobian = np.zeros(jacobian_shape)
    for batch_index in range(batch_size):
        softmax_values = softmax(batch[batch_index])
        for i in range(input_size):
            for j in range(input_size):
                if i == j:
                    value = (1 - softmax_values[i]) * softmax_values[i]
                else:
                    value = -softmax_values[i] * softmax_values[j]

                jacobian[batch_index, i, j] = value

    return jacobian.squeeze() if x.ndim == 1 else jacobian


def softmax_jacobian(x: np.ndarray) -> np.ndarray:
    """a numpy optimized softmax jacobian"""
    if x.ndim not in [1, 2]:
        raise ValueError(
            f"Invalid input shape for the jacobian of an element wise function:\n {x}"
        )

    batch = softmax(x)
    if x.ndim == 1:
        batch = np.expand_dims(batch, axis=0)
    input_size = batch.shape[-1]
    diag_terms = batch - batch**2
    jacobian = -batch[:, :, None] * batch[:, None, :]
    diag_indices = np.arange(input_size)
    jacobian[:, diag_indices, diag_indices] = diag_terms
    return jacobian.squeeze() if x.ndim == 1 else jacobian
