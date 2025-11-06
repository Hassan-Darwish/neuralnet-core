from src.activations.activation import Activation
from numpy.typing import NDArray
import numpy as np

class SoftMax(Activation):
    """Implements the SoftMax activation function.
    
    Converts a vector of logits into a probability distribution.
    """
    
    def __init__(self) -> None:
        """Initializes the SoftMax activation.
        
        It doesn't need any parameters.
        """
        super().__init__()


    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Performs the stable softmax forward pass."""

        # self.inputs is already stored in the parent 'Activation' class,
        # but storing it here is fine if you prefer.
        self.inputs = inputs
        
        # The 'inputs' *are* the logits from the previous layer.
        exp_logits = np.exp(inputs - np.max(inputs, axis=1, keepdims=True), dtype=np.float32)
        
        self.output = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(np.float32)
        return self.output
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Performs the backward pass for the Softmax activation.
        (Your logic here was already correct!)
        """
    
            
        # 1. Calculate the dot product (this was correct)
        dot_product = np.sum(grad_output * self.output, axis=1, keepdims=True)
        
        # 2. Apply the vectorized formula (this was also correct)
        grad_input = self.output * (grad_output - dot_product)
        
        return grad_input.astype(np.float32)