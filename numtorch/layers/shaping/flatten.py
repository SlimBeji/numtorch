import numpy as np

from numtorch.layers.base import BaseLayer


class Flatten(BaseLayer):
    IS_TRAINABLE = False

    def __init__(self, ndim: int):
        if ndim < 2:
            raise ValueError(
                f"ndim must be at least equal to 2, received {ndim} instead"
            )

        super().__init__(trainable=False)
        self.ndim = ndim

    def _check_input(self, x: np.ndarray) -> np.ndarray:
        """Accept only np.ndarray with two dimensions"""
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Expected input of type np.ndarray, received {type(x)} instead"
            )

        if x.ndim not in [self.ndim, self.ndim + 1]:
            raise ValueError(
                f"Expected a {self.ndim}D input or a {self.ndim + 1}D batch, received {x.ndim}D instead"
            )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == self.ndim:
            # Received a single input
            return x.flatten()
        else:
            # received a batch
            return x.reshape(x.shape[0], -1)

    def _grad(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == self.ndim:
            # Received a single input
            output_size = np.prod(x.shape)
            identity: np.ndarray = np.eye(output_size)
            return identity.reshape(output_size, *x.shape)
        else:
            # received a batch
            batch_size = x.shape[0]
            output_size = np.prod(x.shape[1:])
            identity: np.ndarray = np.eye(output_size)
            single_jacobian = identity.reshape(output_size, *x.shape[1:])
            return np.stack([single_jacobian] * batch_size, axis=0)
