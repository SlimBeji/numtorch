import numpy as np

from numtorch.layers.base import BaseLayer, param_grad


class LinearLayer(BaseLayer):
    IS_TRAINABLE = True

    def __init__(self, in_feature: int, out_feature: int):
        super().__init__(trainable=True)
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weights = np.random.randn(out_feature, in_feature)
        self.bias = np.zeros(out_feature)

    def _check_input(self, x: np.ndarray):
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"Expected input of type np.ndarray, received {type(x)} instead"
            )

        if x.ndim == 0:
            raise ValueError(f"Linear Layers does not accept scalars, received {x}")
        if x.ndim > 2:
            raise ValueError(
                f"Linear Layers does not accept tensors of dimension bigger than 2, received shape {x}"
            )

        if x.shape[-1] != self.in_feature:
            mismatch = "Size mismatch: Linear Layer in_features: {} different from {}"
            raise ValueError(mismatch.format(self.in_feature, x.shape[-1]))

    def _forward(self, x: np.ndarray) -> np.ndarray:
        return np.matmul(x, self.weights.T) + self.bias

    def _grad(self, x: np.ndarray) -> np.ndarray:
        """Derivative of AX + b is just A"""
        result = self.weights
        if x.ndim == 2:  # x is a batch of vectors
            result = np.tile(result, (x.shape[0], 1, 1))
        return result

    @param_grad("weights")
    def _weights_grad(self, x: np.ndarray) -> np.ndarray:
        """[Df/DW]ijk = Df_k/Dw_ij"""
        # Converting to batch format
        if x.ndim == 1:  # convert vector to batch
            batch = x[None, :]
        else:
            batch = x

        batch = np.tile(
            batch[:, None, None, :], (1, self.out_feature, self.out_feature, 1)
        )
        result = np.zeros(
            (batch.shape[0], self.out_feature, self.out_feature, self.in_feature)
        )
        i_indices = np.arange(self.out_feature)
        result[:, i_indices, i_indices, :] = batch[:, i_indices, i_indices, :]

        # remove extra dimension if x is not a batch
        if x.ndim == 1:
            result = result[0]

        return result

    @param_grad("bias")
    def _bias_grad(self, x: np.ndarray) -> np.ndarray:
        """[Df/Db]ij = Df_i/Db_j"""
        result = np.eye(self.out_feature)
        if x.ndim == 2:  # x is a batch of vectors
            result = np.tile(result, (x.shape[0], 1, 1))
        return result
