import numpy as np

from layers.base import BaseLayer

class ReluActivation(BaseLayer):
    def _forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
