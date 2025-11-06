from src.activations.activation import Activation
from numpy.typing import NDArray
import numpy as np

class Linear(Activation):
    """Implements the Linear activation function (identity).
    
    Formula: f(x) = x
    """
    def __init__(self) -> None:
        """Initializes the Linear activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
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
        if self.inputs is not None and self.outputs is not None: # type: ignore
            # Local gradient is 1
            return grad_output.astype(np.float32)
        raise ValueError('Error: backward method was called before forward method')

