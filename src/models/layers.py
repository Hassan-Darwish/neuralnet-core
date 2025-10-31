"""Defines the abstract base class for a neural network layer.

This module provides the `Layer` abstract class, which serves as an
interface for all neural network layers (e.Example: Dense, Conv2D,
Activation). It defines the essential methods required for forward and
backward propagation.

Author: Hassan Darwish
Date: October 2025
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from numpy.typing import NDArray


class Layer(ABC):
    """Abstract base class for a neural network layer.

    All layers must inherit from this class and implement the `forward`
    and `backward` methods.

    Attributes:
        trainable (bool): True if the layer has parameters that can be
                          updated during training.
        initialized (bool): True if the layer parameters are set in the
                            forward function.
        params (Dict[str, Any]): A dictionary holding the layer's trainable
                                 parameters (e.g., 'weights', 'biases').
        grads (Dict[str, Any]): A dictionary holding the gradients computed
                                during backpropagation, corresponding to
                                the keys in `params`.
        input (Optional[Any]): Stores the input received during the
                               forward pass, often needed for the
                               backward pass.
        output (Optional[Any]): Stores the output computed during the
                                forward pass.
    """

    def __init__(self) -> None:
        """Initializes the base layer attributes."""
        self.trainable: bool = False
        self.initialized: bool = False
        self.params: Dict[str, NDArray[np.float32]] = {}
        self.grads: Dict[str, NDArray[np.float32]] = {}
        self.input: Optional[NDArray[np.float32]] = None
        self.output: Optional[NDArray[np.float32]] = None

    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Performs the forward pass of the layer.

        Args:
            inputs: The data or activations from the previous layer.

        Returns:
            The computed output of this layer.
        """
        pass

    @abstractmethod
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the backward pass of the layer.

        Computes the gradient of the loss with respect to the layer's
        inputs and stores gradients for any trainable parameters.

        Args:
            grad_output: The gradient of the loss with respect to this
                         layer's output (computed by the next layer).

        Returns:
            The gradient of the loss with respect to this layer's input.
        """
        pass

    def update(self, optimizer: Optional[Any] = None) -> None:
        """Updates the layer's parameters using an optimizer.

        This method is typically only implemented by trainable layers
        (e.g., Dense, Conv2D).

        Args:
            optimizer: An optimizer object (e.g., SGD, Adam) that
                       knows how to update parameters using the
                       computed gradients.
        """
        # Default implementation does nothing, as non-trainable
        # layers have no parameters to update.
        pass