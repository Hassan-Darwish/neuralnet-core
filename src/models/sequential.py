from src.models.model import Model
from typing import List, Optional
from numpy.typing import NDArray
from src.optimizers.optimizer import Optimizer
from src.losses.loss import Loss
import numpy as np
import src.layers as layer
import src.optimizers as optimizers
import src.losses as losses

class Sequential(Model):
    
    def __init__(self, layers: List[layer.Layer]) -> None:
        self.time: int = 0
        self.layers = layers
        self.optimizer: Optional[Optimizer] = None
        self.loss: Optional[Loss] = None

    def add(self, layer: layer.Layer, layers: Optional[List[layer.Layer]]) -> None:
        self.layers.append(layer)

        if layers is not None:
            for l in layers:
                self.layers.append(l)

    def pop(self) -> None:
        self.layers.pop()

    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        output :NDArray[np.float32] = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output
    
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        output: NDArray[np.float32] = grad_output

        for layer in reversed(self.layers):
            output = layer.backward(output)

        return output

    def compile(self, optimizer: str, loss: str, **kwargs: float) -> None:
        optimizer_instance: Optional[Optimizer] = optimizers.get_optimizer(optimizer, **kwargs)
        loss_instance: Optional[Loss] = losses.get_loss(loss)

        self.optimizer = optimizer_instance
        self.loss = loss_instance
    
    def fit(
        self,
        true_inputs: NDArray[np.float32],
        true_outputs: NDArray[np.float32],
        epochs: int,
        batch_size: Optional[int]
        ) -> None:
    
        for epoch in range(epochs):
        
            pred_outputs = self.forward(inputs=true_inputs)

            if self.loss is None or self.optimizer is None:
                raise ValueError("Error: fit called before compile.")                
            
            loss = self.loss.forward(y_pred=pred_outputs, y_true=true_outputs)

            initial_grad = self.loss.backward(y_pred=pred_outputs, y_true=true_outputs)

            self.backward(initial_grad)

            self._optimizer_step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def _optimizer_step(self):
        """
        Performs a single parameter update step across all layers.
        """
        # 1. Check if an optimizer is even configured
        if self.optimizer is None:
            return
            
        # 2. Increment the timestep ONCE per training step
        self.time += 1

        # 3. Loop through all layers
        for layer in self.layers:
            # 4. Only update layers that are trainable (have params)
            if layer.trainable:
                
                # 5. Loop through all params in that layer (e.g., 'W' and 'b')
                for param_name in layer.params:
                    
                    # 6. Call the optimizer to get the *new* parameters
                    updated_params = self.optimizer.update(
                        params=layer.params[param_name],
                        grads=layer.grads[param_name],
                        time=self.time  # Pass the correct, global timestep
                    )
                    
                    # 7. IMPORTANT: Re-assign the new params back to the layer
                    layer.params[param_name] = updated_params