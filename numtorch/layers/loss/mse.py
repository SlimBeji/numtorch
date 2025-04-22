import numpy as np

from numtorch.layers.loss.base import LossLayer
from numtorch.utils.loss import mse


class MSELoss(LossLayer):
    IS_TRAINABLE = False

    def _check_input(self, y: np.ndarray, target: np.ndarray):
        if not isinstance(y, np.ndarray) or not isinstance(target, np.ndarray):
            raise TypeError(
                "Expected inputs of type np.ndarray, received ({}, {}) instead".format(
                    type(y), type(target)
                )
            )

        if y.shape != target.shape:
            raise ValueError("Shape mismatch between preds and targets")

    def _forward(self, y: np.ndarray, target: np.ndarray) -> float:
        return mse(y, target)

    def _grad(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 2 * (y - target) / y.size
