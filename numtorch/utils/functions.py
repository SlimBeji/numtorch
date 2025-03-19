import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_jacobian(x: np.ndarray) -> np.ndarray:
    return np.eye(len(x)) * sigmoid_prime(x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def tanh_jacobian(x: np.ndarray) -> np.ndarray:
    return np.eye(len(x)) * tanh_prime(x)

def softmax(x: np.ndarray) -> np.ndarray:
    maximums = x.max(axis=-1 , keepdims=True)
    shifted = x - maximums
    exp: np.ndarray = np.exp(shifted)
    exp_sums = exp.sum(axis=-1, keepdims=True)
    return exp / exp_sums

def softmax_jacobian(x: np.ndarray) -> np.ndarray:
    f = softmax(x)
    n = x.shape[-1]
    shape = [*x.shape[:-1], n, n]
    result = np.zeros(shape)
    for i in range(n):
        for j in range(n):
            if i == j:
                result[...,i, j] =  f[..., i] * (1 - f[..., i])
            else:
                result[...,i, j] = -f[..., i] * f[..., j]
    return result
