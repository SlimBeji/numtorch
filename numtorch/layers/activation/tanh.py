import numpy as np

from numtorch.layers.base import BaseLayer
from numtorch.utils import tanh, tanh_jacobian


class TanhActivation(BaseLayer):
    def _forward(self, x: np.ndarray) -> np.ndarray:
        return tanh(x)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        return tanh_jacobian
