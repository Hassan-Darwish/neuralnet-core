"""Defines activation functions and a factory to retrieve them.

This module provides an abstract base class `Activations` and several
concrete implementations (Sigmoid, ReLU, Linear). It also includes a
helper function `get_activation` to instantiate activation
objects from string names.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union
from abc import ABC, abstractmethod

class Activations(ABC):
    """Abstract base class for all activation functions."""
    
    def __init__(self) -> None:
        """Initializes the base activation attributes.
        
        Attributes:
            inputs (Optional[NDArray[np.float32]]): Stores the input array
                from the forward pass, required for backpropagation.
            outputs (Optional[NDArray[np.float32]]): Stores the computed
                output array from the forward pass.
        """
        self.inputs: Optional[NDArray[np.float32]] = None
        self.outputs: Optional[NDArray[np.float32]] = None

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Performs the forward pass of the activation function.

        Args:
            inputs (NDArray[np.float32]): The input data (output of the
                                          previous layer).

        Returns:
            Optional[NDArray[np.float32]]: The transformed data after
                                           applying the activation.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the backward pass (backpropagation).

        Computes the gradient of the loss with respect to the activation's
        input.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the loss
                                               with respect to this
                                               activation's output.

        Returns:
            NDArray[np.float32]: The gradient of the loss with respect to
                                 this activation's input.
        """
        pass

class Sigmoid(Activations):
    """Implements the Sigmoid activation function.
    
    Formula: f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self) -> None:
        """Initializes the Sigmoid activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Applies the sigmoid function element-wise.

        Args:
            inputs (NDArray[np.float32]): The input data.

        Returns:
            Optional[NDArray[np.float32]]: The activated output, with values
                                           between 0 and 1.
        """
        self.inputs = inputs
        # f(x) = 1 / (1 + exp(-x))
        self.outputs = (1 / (1 + np.exp(-self.inputs))).astype(np.float32)
        return self.outputs
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Computes the gradient of the sigmoid function.

        Gradient formula: dL/dX = dL/dOut * f'(x)
                          where f'(x) = f(x) * (1 - f(x))

        Args:
            grad_output (NDArray[np.float32]): Gradient from the next layer.

        Returns:
            NDArray[np.float32]: The gradient to pass to the previous layer.
        
        Raises:
            ValueError: If `backward` is called before `forward`.
        """
        if self.inputs is not None and self.outputs is not None:
            # Local gradient is self.outputs * (1 - self.outputs)
            return (grad_output * self.outputs * (1 - self.outputs)).astype(np.float32)
        raise ValueError('Error: backward method was called before forward method')
    
class ReLU(Activations):
    """Implements the Rectified Linear Unit (ReLU) activation function.
    
    Formula: f(x) = max(0, x)
    """
    def __init__(self) ->None:
        """Initializes the ReLU activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Applies the ReLU function element-wise.

        Args:
            inputs (NDArray[np.float32]): The input data.

        Returns:
            Optional[NDArray[np.float32]]: The activated output.
        """
        self.inputs = inputs
        # f(x) = max(0, x)
        self.outputs = (np.maximum(0.0 ,inputs)).astype(np.float32)
        return self.outputs 
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Computes the gradient of the ReLU function.

        Gradient formula: dL/dX = dL/dOut * f'(x)
                          where f'(x) = 1 if x > 0, else 0

        Args:
            grad_output (NDArray[np.float32]): Gradient from the next layer.

        Returns:
            NDArray[np.float32]: The gradient to pass to the previous layer.
        
        Raises:
            ValueError: If `backward` is called before `forward`.
        """
        if self.inputs is not None and self.outputs is not None:
            # Local gradient is 1 for positive inputs, 0 otherwise
            return (grad_output * (self.inputs > 0))
        raise ValueError('Error: backward method was called before forward method')
    
class Linear(Activations):
    """Implements the Linear activation function (identity).
    
    Formula: f(x) = x
    """
    def __init__(self) -> None:
        """Initializes the Linear activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Applies the linear (identity) function.

        Args:
            inputs (NDArray[np.float32]): The input data.

        Returns:
            Optional[NDArray[np.float32]]: The unchanged input data.
        """
        self.inputs = inputs
        self.outputs = inputs
        return self.outputs
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Computes the gradient of the linear function.

        Gradient formula: dL/dX = dL/dOut * f'(x)
                          where f'(x) = 1

        Args:
            grad_output (NDArray[np.float32]): Gradient from the next layer.

        Returns:
            NDArray[np.float32]: The unchanged gradient from the next layer.
        
        Raises:
            ValueError: If `backward` is called before `forward`.
        """
        if self.inputs is not None and self.outputs is not None:
            # Local gradient is 1
            return grad_output.astype(np.float32)
        raise ValueError('Error: backward method was called before forward method')

    

# map lower-case names to activation classes
_ACTIVATION_MAP: dict[str, type] = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "linear": Linear,
}

def get_activation(
    activation: Optional[Union[Activations, str]]
) -> Optional[Activations]:
    """Factory function to retrieve an activation function instance.

    This helper function provides flexibility by allowing activations to be
    specified as `None`, an existing `Activations` object, or a string name.

    Args:
        activation (Optional[Union[Activations, str]]):
            - None: Returns None.
            - Activations instance: Returns the instance unchanged.
            - str: Returns a new instance of the matching class from
                   `_ACTIVATION_MAP`.

    Returns:
        Optional[Activations]: An instance of an activation class or None.

    Raises:
        ValueError: If the provided string name is not in `_ACTIVATION_MAP`.
        TypeError: If `activation` is not None, an `Activations`
                   instance, or a string.
    """
    if activation is None:
        return None
    if isinstance(activation, Activations):
        return activation
    if isinstance(activation, str): # type: ignore
        key = activation.strip().lower()
        try:
            cls = _ACTIVATION_MAP[key]
        except KeyError:
            raise ValueError(f"Unknown activation '{activation}'. Available: {list(_ACTIVATION_MAP)}")
        return cls()  # instantiate and return
    raise TypeError("activation must be None, an Activations instance or a string name")
