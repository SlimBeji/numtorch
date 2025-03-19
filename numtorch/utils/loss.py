import numpy as np 

from utils.functions import sigmoid

def mse(y: np.ndarray, target: np.ndarray) -> float:
    return ((y - target) ** 2).mean()


def bce_with_logits(logits: np.ndarray, target: np.ndarray) -> float:
    loss = (1 - target) * logits + np.log(1 + np.exp(-logits))
    return loss.mean()

def bce_with_logits_jacobian(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (sigmoid(logits) - target) / logits.size
