from src.activations.activation import Activation
from numpy.typing import NDArray
import numpy as np

class Sigmoid(Activation):
    """Implements the Sigmoid activation function.
    
    Formula: f(x) = 1 / (1 + exp(-x))
    """

    def __init__(self) -> None:
        """Initializes the Sigmoid activation."""
        super().__init__()

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
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
        if self.inputs is not None and self.outputs is not None: # type: ignore
            # Local gradient is self.outputs * (1 - self.outputs)
            return (grad_output * self.outputs * (1 - self.outputs)).astype(np.float32)
        raise ValueError('Error: backward method was called before forward method')