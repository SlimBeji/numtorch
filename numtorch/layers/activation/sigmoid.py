import numpy as np

from numtorch.layers.base import BaseLayer
from numtorch.utils import sigmoid, sigmoid_jacobian


class SigmoidActivation(BaseLayer):
    def _forward(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return sigmoid_jacobian(x)
