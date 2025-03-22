import numpy as np

from numtorch.layers.loss.base import LossLayer
from numtorch.utils.loss import mse


class MSELoss(LossLayer):
    def _forward(self, y: np.ndarray, target: np.ndarray) -> float:
        return mse(y, target) / 2

    def _grad(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        return (y - target).mean()
