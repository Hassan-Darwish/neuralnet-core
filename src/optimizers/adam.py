from src.optimizers.optimizer import Optimizer
from numpy.typing import NDArray
from typing import Dict
import numpy as np

class Adam(Optimizer):
    def __init__(self,
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, NDArray[np.float32]] = {}
        self.v: Dict[int, NDArray[np.float32]] = {}

    def update(
            self, 
            params: NDArray[np.float32],
            grads: NDArray[np.float32],
            time: int
            ) -> NDArray[np.float32]:  
        pid = id(params)

        if pid not in self.m:
            self.m[pid] = np.zeros_like(params, dtype=np.float32)
            self.v[pid] = np.zeros_like(params, dtype=np.float32)

        self.m[pid] = (self.beta1 * self.m[pid] + (1 - self.beta1) * grads).astype(np.float32)
        self.v[pid] = (self.beta2 * self.v[pid] + (1 - self.beta2) * (grads ** 2)).astype(np.float32)

        m_hat = self.m[pid] / (1 - self.beta1 ** time)
        v_hat = self.v[pid] / (1 - self.beta2 ** time)

        updated_params = (params - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.eps))).astype(np.float32)
        return updated_params