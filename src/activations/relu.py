from src.activations.activation import Activation
from numpy.typing import NDArray
import numpy as np

class ReLU(Activation):
    """Implements the Rectified Linear Unit (ReLU) activation function.
    
    Formula: f(x) = max(0, x)
    """
    def __init__(self) ->None:
        """Initializes the ReLU activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
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
        if self.inputs is not None and self.outputs is not None: # type: ignore
            # Local gradient is 1 for positive inputs, 0 otherwise
            return (grad_output * (self.inputs > 0))
        raise ValueError('Error: backward method was called before forward method')