import numpy as np

from numtorch.layers.base import BaseLayer


class ReluActivation(BaseLayer):
    def _check_input(self, x: np.ndarray) -> np.ndarray:
        """Accept all sort of inputs, provided they are
        of type np.ndarray"""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Expected input of type np.ndarray, received {type(x)} instead"
            )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
