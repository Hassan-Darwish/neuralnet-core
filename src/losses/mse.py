from src.losses.loss import Loss
from numpy.typing import NDArray
import numpy as np

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