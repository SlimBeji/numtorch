from optimizers.base import BaseOptimizer
import numpy as np

class SGD(BaseOptimizer):
    def __init__(self, model, lr: float = 0.001, weight_decay: float = 0):
        super().__init__(model)
        self.lr = lr
        self.weight_decay = weight_decay

    def _step(self, param: np.ndarray, loss_grad: np.ndarray) -> np.ndarray:
        update = (loss_grad + self.weight_decay * param) * self.lr
        return param - update
