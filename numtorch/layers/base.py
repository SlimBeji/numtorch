from typing import Callable

import numpy as np

from config import Config


def param_grad(name: str) -> Callable:
    def wrapper(f: Callable) -> Callable:
        f.is_parameter_grad = True
        f.parameter_name = name
        return f

    return wrapper


class BaseLayer:
    IS_TRAINABLE = False

    def __init__(self, trainable: bool = False):
        self._name = ""
        self._trainable: bool = trainable and self.IS_TRAINABLE
        self.inputs_grad: np.ndarray | None
        self.parameters_grad: dict[str, np.ndarray] = {}
        self.parameters_loss_grad: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, layer_name: str):
        self._name = layer_name

    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, val: bool):
        self._trainable = val and self.IS_TRAINABLE

    def __call__(self, *args: tuple[np.ndarray]) -> np.ndarray:
        """Implementation of the forward pass.
        Each time we do the forward pass, we want to
        compute the layer output and layer gradient output
        """
        # Step 1: Check if the input validity
        self._check_input(*args)

        # Step 2: Do the forward pass
        result = self._forward(*args)

        # Step 3: Compute the grad if we are not in evaluation mode
        if not Config.NUMTORCH_EVAL:
            self.inputs_grad = self._grad(*args)
        else:
            self.inputs_grad = None

        # Step 4: Compute the grad with respect of the parameters
        if self.trainable and not Config.NUMTORCH_EVAL:
            self.parameters_grad = self._training_grad(*args)
        else:
            self.parameters_grad = {}

        return result

    def _check_input(self, *args: tuple[np.ndarray]):
        pass

    def _forward(self, *args: tuple[np.ndarray]) -> np.array:
        """Do the forward pass on the layer"""
        raise NotImplementedError

    def _grad(self, *args: tuple[np.ndarray]) -> np.array:
        """Compute the layer gradient with respect to the inputs.
        Doing like a forward pass but on gradient of the layer"""
        raise NotImplementedError

    def _training_grad(self, *args: tuple[np.ndarray]) -> dict[str, np.array]:
        """Compute the layer gradient with respect to its parameters.
        Returns by default empty dict because some layers (e.g. activation layers)
        does not have parameters"""
        result = dict()
        for name in dir(self):
            attr = getattr(self, name, None)
            is_parameter_grad: bool = getattr(attr, "is_parameter_grad", False)
            parameter_name: str = getattr(attr, "parameter_name", "")

            if not parameter_name or not is_parameter_grad:
                # We go to the next property
                continue

            if getattr(self, parameter_name, None) is None:
                raise ValueError(
                    f"parameter {parameter_name} with grad method was not defind"
                )

            result[parameter_name] = attr(*args)

        return result

    def _loss_grad(self, cumulated_grad: np.ndarray):
        for name, param_grad in self.parameters_grad.items():
            loss_grad = np.matmul(param_grad, cumulated_grad)
            self.parameters_loss_grad[name] = loss_grad
