import numpy as np

from numtorch.layers.base import BaseLayer
from numtorch.utils import softmax, softmax_jacobian


class SoftmaxActivation(BaseLayer):
    def _forward(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return softmax_jacobian(x)
