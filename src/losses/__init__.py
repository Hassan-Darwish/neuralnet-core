from .loss import Loss
from .mse import MeanSquaredError
from .cce import CategoricalCrossEntropy
from typing import Optional

def get_loss(name: Optional[str]) -> Optional[Loss]:
    if name is None:
        return None
    
    name = name.lower()

    if name == "mse":
        return MeanSquaredError()
    elif name == "cce":
        return CategoricalCrossEntropy()
    else:
        raise ValueError(f"Loss Error: Unknown loss function {name}")