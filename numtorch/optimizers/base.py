from typing import TYPE_CHECKING

import numpy as np
from layers.base import BaseLayer

if TYPE_CHECKING:
    from models.base import BaseModel


class BaseOptimizer:
    def __init__(self, model: "BaseModel"):
        self.model = model

    def zero_grad(self):
        for layer in self.model.layers:
            layer: BaseLayer
            layer.parameters_grad = {}

    def step(self):
        """Update all the model layers"""
        for layer in self.model.layers:
            layer: BaseLayer
            for name, loss_grad in layer.parameters_loss_grad.items():
                param = getattr(layer, name, None)
                if param is None:
                    raise RuntimeError(
                        f"Could not find param {name} for layer {layer.name}"
                    )

                updated = self._step(param, loss_grad)
                setattr(layer, name, updated)

    def _step(self, param: np.ndarray, loss_grad: np.ndarray) -> np.ndarray:
        """update a parameter"""
        raise NotImplementedError
