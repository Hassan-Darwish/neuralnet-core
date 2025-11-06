from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np


class Optimizer(ABC):

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    @abstractmethod
    def update(
            self, 
            params: NDArray[np.float32],
            grads: NDArray[np.float32],
            time: int
            ) -> NDArray[np.float32]:
        
        raise NotImplementedError