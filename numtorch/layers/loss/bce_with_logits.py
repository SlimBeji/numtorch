import numpy as np

from numtorch.layers.loss.base import LossLayer
from numtorch.utils.loss import bce_with_logits, bce_with_logits_jacobian


class BCEWithLogitsLoss(LossLayer):
    def _forward(self, y: np.ndarray, target: np.ndarray) -> float:
        return bce_with_logits(y, target)

    def _grad(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        return bce_with_logits_jacobian(y, target)
