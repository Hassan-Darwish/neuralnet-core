"""Implements a fully connected (Dense) neural network layer.

This module provides the `Dense` class, which is a concrete implementation
of the abstract `Layer` class. It performs a linear operation
(matrix multiplication and optional bias addition).

Author: Hassan Darwish
Date: October 2025
"""

from src.layers.layer import Layer
from numpy.typing import NDArray
from typing import Optional
import src.activations as activations
import numpy as np

class Dense(Layer):
    """A fully connected (dense) layer.

    This layer implements the operation:
    `output = dot(input, kernel) + bias`

    The weights (kernel) and biases are initialized lazily during the
    first `forward()` call, as their size depends on the input shape.

    Attributes:
        units (int): The number of output neurons (dimensionality of the
                     output space).
        activation (Optional[str]): Name of the activation function to use.
                                    (Note: Activation logic is not
                                    implemented in this class).
        use_bias (bool): Whether the layer uses a bias vector.
        kernel_initializer (str): The initialization method to use for
                                  the layer's weights.
        initialized (bool): Tracks if the layer's parameters have been
                            initialized.
    """

    # --- Constants for clarity in array indexing ---
    BATCH_INDEX = 0
    """Axis index for the batch size (samples)."""

    FEATURE_INDEX = 1
    """Axis index for the features (input dimensions)."""

    ONE_ROW = 1
    """Used for creating bias vectors with the correct broadcastable shape."""

    COLUMN_AXIS = 0
    """Axis to sum over when computing the bias gradient."""

    def __init__(self,
                 units: int,
                 activation: Optional[str] = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'he_normal') -> None:
        """Initializes the Dense layer configuration.

        Args:
            units (int): The number of output neurons.
            activation (Optional[str]): Name of activation function (e.g.,
                                        'relu', 'sigmoid'). Stored but not
                                        applied by this layer.
            use_bias (bool): Whether to include a learnable bias term.
            kernel_initializer (str): Method for initializing weights.
                                      Supports 'he' and 'xavier'.
        """
        super().__init__()
        self.trainable = True
        self.units = units
        self.activation = activations.get_activation(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.initialized: bool = False

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the forward pass and lazy initialization.

        On the first call, this method initializes the layer's weights
        and biases based on the input shape.

        Args:
            inputs (NDArray[np.float32]): The input data from the previous
                                          layer, with shape
                                          (batch_size, input_features).

        Returns:
            Optional[NDArray[np.float32]]: The layer's output, with shape
                                           (batch_size, self.units).
        """
        # Lazy initialization of parameters on the first forward pass
        if not self.initialized:
            # Infer input features from the input shape
            fan_in: int = inputs.shape[self.FEATURE_INDEX]
            
            # Initialize weights and gradients
            self.params['W'] = self._init_weights(fan_in, self.units, self.kernel_initializer)
            self.grads['W'] = np.zeros_like(self.params['W'])
            
            # Initialize bias and its gradient if use_bias is True
            if self.use_bias:
                self.params['b'] = np.zeros((self.ONE_ROW, self.units), dtype=np.float32)
                self.grads['b'] = np.zeros_like(self.params['b'])
            
            self.initialized = True

        # Store input for the backward pass
        self.input = inputs
        
        # Add bias if configured
        if self.use_bias:
            z: NDArray[np.float32] = np.dot(self.input, self.params['W']) + self.params['b']
        else:
            z: NDArray[np.float32] = np.dot(self.input, self.params['W'])

        # add activation if configured
        if self.activation is not None:
            self.output = self.activation.forward(z)
        else:
            self.output = z

        return self.output
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the backward pass (backpropagation).

        Computes the gradients of the loss with respect to the layer's
        parameters (W, b) and its input.

        Args:
            grad_output (NDArray[np.float32]): The gradient of the loss
                                               with respect to this
                                               layer's output.

        Returns:
            NDArray[np.float32]: The gradient of the loss with respect to
                                 this layer's input.
        """

        # add activation if configured
        if self.activation is not None:
            grad_output = self.activation.backward(grad_output)

        # Gradient of loss w.r.t. weights (dW)
        # dL/dW = dL/dOut * dOut/dW = X.T @ grad_output
        self.grads['W'] = np.dot(self.input.T, grad_output)
        
        # Gradient of loss w.r.t. bias (db)
        if self.use_bias:
            # dL/db = dL/dOut * dOut/db = sum(grad_output)
            self.grads['b'] = np.sum(grad_output, axis=self.COLUMN_AXIS, keepdims=True)

        # Gradient of loss w.r.t. input (dX)
        # dL/dX = dL/dOut * dOut/dX = grad_output @ W.T
        grad_input = np.dot(grad_output, self.params['W'].T)
        
        return grad_input

    def _init_weights(self, 
                      fan_in: int, 
                      fan_out: int,
                      method: str = 'he_normal') -> NDArray[np.float32]:
        """Initializes the weight matrix based on the specified method.

        Args:
            fan_in (int): The number of input features.
            fan_out (int): The number of output features.
            method (str): The name of the initialization method ('he_normal' or
                          'glorot_uniform'), defaulted to 'he_normal'.

        Returns:
            NDArray[np.float32]: The initialized weight matrix, cast to
                                 float32.

        Raises:
            ValueError: If an unknown initialization method is specified.
        """
        method_lower = method.lower()
        
        if method_lower == 'he_normal':
            # He initialization (good for ReLU)
            # std = sqrt(2 / fan_in)
            scale = np.sqrt(2.0 / fan_in)
        elif method_lower == 'glorot_uniform':
            # Xavier/Glorot initialization (good for tanh)
            # std = sqrt(2 / (fan_in + fan_out))
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unknown weight initializiation method = {method}")
            
        # Generate weights from a standard normal distribution and scale
        # Cast to float32 to match the type hints and save memory
        return (np.random.randn(fan_in, fan_out) * scale).astype(np.float32)

