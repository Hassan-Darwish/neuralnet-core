from .activation import Activation
from .sigmoid import Sigmoid
from .relu import ReLU
from .linear import Linear
from .softmax import SoftMax
from typing import Optional

def get_activation(name: Optional[str]) -> Optional[Activation]:
    
    if name is None:
        return None
    
    name = name.lower()

    if name == "sigmoid":
        return Sigmoid()
    elif name == "relu":
        return ReLU()
    elif name == "linear":
        return Linear()
    elif name == "softmax":
        return SoftMax()
    else:
        raise ValueError(f"Activation Error: unknown activation function {name}")