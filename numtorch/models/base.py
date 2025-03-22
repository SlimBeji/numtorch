import numpy as np

from numtorch.layers.base import BaseLayer


class BaseModel:
    def __init__(self, *args, **kwargs):
        self.layers = self.build(*args, **kwargs)

    def build(self) -> list[BaseLayer]:
        return []

    def __call__(self, *args: tuple[np.ndarray]) -> np.ndarray:
        """Forward pass through all layers"""
        result: np.ndarray = None
        for layer in self.layers:
            if result is None:
                result = layer(*args)
            else:
                result = layer(result)
        return result
