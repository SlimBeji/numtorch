import numpy as np

from numtorch.layers.base import BaseLayer
from numtorch.utils import sigmoid, sigmoid_jacobian


class SigmoidActivation(BaseLayer):
    IS_TRAINABLE = False

    def _check_input(self, x: np.ndarray) -> np.ndarray:
        """Accept all sort of inputs, provided they are
        of type np.ndarray"""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Expected input of type np.ndarray, received {type(x)} instead"
            )

        if x.ndim not in [1, 2]:
            raise ValueError(f"Expected a 1D or 2D input, received {x.ndim}D instead")

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return sigmoid_jacobian(x)
