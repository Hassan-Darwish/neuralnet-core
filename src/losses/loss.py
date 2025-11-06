"""Defines loss functions and their derivatives.

This module provides an abstract base class `Loss` and concrete
implementations for common loss functions used in training neural
networks, such as Mean Squared Error and Cross-Entropy.

Author: Hassan Darwish
Date: November 2025
"""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np

class Loss(ABC):
    """Abstract base class for all loss functions."""

    AXIS_SHAPE = 0
    """Index to get the batch size from a shape tuple."""

    @abstractmethod
    def forward(
            self, 
            y_pred: NDArray[np.float32], 
            y_true: NDArray[np.float32]
            ) -> np.float32:
        """Calculates the forward pass of the loss function.

        Args:
            y_pred (NDArray[np.float32]): The predictions from the model.
            y_true (NDArray[np.float32]): The true target values.

        Returns:
            np.float32: The mean loss value over the batch.
        """
        raise NotImplementedError
    
    @abstractmethod
    def backward(
            self, 
            y_pred: NDArray[np.float32],
            y_true: NDArray[np.float32]
            ) -> NDArray[np.float32]:
        """Calculates the gradient of the loss with respect to the predictions.

        Args:
            y_pred (NDArray[np.float32]): The predictions from the model.
            y_true (NDArray[np.float32]): The true target values.

        Returns:
            NDArray[np.float32]: The gradient of the loss (dL/dy_pred).
        """
        raise NotImplementedError
    
class MeanSquaredError(Loss):
    """Calculates the Mean Squared Error (MSE) loss.
    
    Formula: L = (1/N) * sum((y_pred - y_true)^2)
    """
    
    def forward(
            self, 
            y_pred: NDArray[np.float32], 
            y_true: NDArray[np.float32]
            ) -> np.float32:
        """Calculates the mean squared error between predictions and targets.

        Args:
            y_pred (NDArray[np.float32]): The predictions from the model.
            y_true (NDArray[np.float32]): The true target values.

        Returns:
            np.float32: The mean squared error.
        """
        # (y_pred - y_true)^2
        squared_error = (y_pred - y_true) ** 2
        # np.mean averages over all elements, giving a single scalar loss
        return np.mean(squared_error, dtype= np.float32)
    
    def backward(
            self, 
            y_pred: NDArray[np.float32],
            y_true: NDArray[np.float32]
            ) -> NDArray[np.float32]:
        """Calculates the gradient of the MSE loss.

        Gradient Formula: dL/dy_pred = (2/N) * (y_pred - y_true)
        
        Args:
            y_pred (NDArray[np.float32]): The predictions from the model.
            y_true (NDArray[np.float32]): The true target values.

        Returns:
            NDArray[np.float32]: The gradient of the MSE loss.
        """
        # Get N (batch size)
        n: NDArray[np.float32] = y_true.shape[self.AXIS_SHAPE]

        # dL/dy_pred = 2 * (y_pred - y_true) / N
        return ((2 * (y_pred - y_true)) / n).astype(np.float32)
    
class CrossEntropy(Loss):
    """Calculates the Categorical Cross-Entropy (CCE) loss.

    This is typically used for multi-class classification problems.
    It expects y_pred to be probabilities (e.g., from a Softmax)
    and y_true to be one-hot encoded vectors.
    
    Formula: L = (-1/N) * sum(y_true * log(y_pred))
    """

    def forward(
            self, 
            y_pred: NDArray[np.float32], 
            y_true: NDArray[np.float32]
            ) -> np.float32:
        """Calculates the mean categorical cross-entropy loss.

        Args:
            y_pred (NDArray[np.float32]): Predicted probabilities (N, C).
            y_true (NDArray[np.float32]): True labels, one-hot (N, C).

        Returns:
            np.float32: The mean CCE loss.
        """
        
        # Epsilon for numerical stability (to avoid log(0))
        eps: float = 1e-9
        
        # Clip predictions to avoid log(0) and log(1) issues
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Calculate loss for each sample: sum(y_true * log(y_pred))
        # Since y_true is one-hot, this just picks the log-prob of the true class
        sample_losses = np.sum(y_true * np.log(y_pred), axis=1)
        
        # Return the mean negative log-likelihood
        return -np.mean(sample_losses, dtype=np.float32)

    
    def backward(
        self, 
        y_pred: NDArray[np.float32],
        y_true: NDArray[np.float32]
        ) -> NDArray[np.float32]:
        """Calculates the gradient of the CCE loss.

        Gradient Formula: dL/dy_pred = (-1/N) * (y_true / y_pred)
        
        Args:
            y_pred (NDArray[np.float32]): Predicted probabilities (N, C).
            y_true (NDArray[np.float32]): True labels, one-hot (N, C).

        Returns:
            NDArray[np.float32]: The gradient of the CCE loss.
        """

        # Get N (batch size)
        n: NDArray[np.float32] = y_true.shape[self.AXIS_SHAPE]
        # Epsilon for numerical stability (to avoid division by zero)
        eps: float = 1e-9

        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # dL/dy_pred = -y_true / (y_pred * N)
        return -(y_true / (y_pred * n)).astype(np.float32)
