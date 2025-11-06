from .optimizer import Optimizer
from .adam import Adam
from typing import Optional

def get_optimizer(name: Optional[str], **kwargs: float) -> Optional[Optimizer]:
    
    if name is None:
        return None
    
    name = name.lower()

    if name == "adam":
        return Adam(**kwargs)
    #extendable
    else: 
        raise ValueError(f"Optimizer Error: Unknown optimizer {name}")