from typing import TYPE_CHECKING

import numpy as np

from numtorch.layers.base import BaseLayer

if TYPE_CHECKING:
    from models.base import BaseModel


class LossLayer(BaseLayer):
    IS_TRAINABLE = False

    def __init__(self, model: "BaseModel"):
        super.__init__(trainable=False)
        self.model = model

    def _forward(self, y: np.ndarray, target: np.ndarray) -> float:
        """Loss layers should return a scalar/float"""
        raise NotImplementedError

    def _grad(self, y: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute the layer gradient with respect to the inputs.
        Doing like a forward pass but on gradient of the layer"""
        raise NotImplementedError

    def backward(self):
        cumulated_grad: np.ndarray = self.inputs_grad
        for layer in reversed(self.model.layers):
            layer: BaseLayer
            if layer.trainable:
                self._loss_grad(cumulated_grad)

            cumulated_grad = np.matmul(layer.inputs_grad, cumulated_grad)
