from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional
import numpy as np

class Model(ABC):
    
    @abstractmethod
    def forward(self, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad_output: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError
    
    @abstractmethod
    def compile(self, optimizer: str, loss: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def fit(
        self,
        true_inputs: NDArray[np.float32],
        true_outputs: NDArray[np.float32],
        epochs: int,
        batch_size: Optional[int]
        ) -> None:
        raise NotImplementedError
        