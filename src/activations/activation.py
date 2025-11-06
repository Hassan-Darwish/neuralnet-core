"""Defines activation functions and a factory to retrieve them.

This module provides an abstract base class `Activation` and several
concrete implementations (Sigmoid, ReLU, Linear). It also includes a
helper function `get_activation` to instantiate activation
objects from string names.

Author: Hassan Darwish
Date: November 2025
"""

from src.layers.layer import Layer
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import numpy as np

class Activation(Layer, ABC):
    """Abstract base class for all activation functions."""
    
    def __init__(self) -> None:
        """Initializes the base activation attributes.
        
        Attributes:
            inputs (Optional[NDArray[np.float32]]): Stores the input array
                from the forward pass, required for backpropagation.
            outputs (Optional[NDArray[np.float32]]): Stores the computed
                output array from the forward pass.
        """
        super().__init__()

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the forward pass of the activation function.

        Args:
            inputs (NDArray[np.float32]): The input data (output of the
                                          previous layer).

        Returns:
            Optional[NDArray[np.float32]]: The transformed data after
                                           applying the activation.
        """
        raise NotImplementedError

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
        raise NotImplementedError

