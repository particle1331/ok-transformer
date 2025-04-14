import torch
import random
import numpy as np


def get_device(): 
    return (
        torch.device("cuda:0") if torch.cuda.is_available() else (
            torch.device("mps") if torch.mps.is_available() else 
                torch.device("cpu")
        )
    )


class eval_context:
    """Context manager to set model to eval mode."""
    def __init__(self, model):
        self.model = model
        self.state = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.train(self.state)


def set_seed(value=42, deterministic=False):
    """Set the seed for reproducibility. 
    
    WARNING: Setting deterministic=True may result in decreased performance. 
    e.g. disabling benchmarking causes cuDNN to deterministically select an 
    algorithm, possibly at the cost of reduced performance.
    """

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    print(f"seed: {value}  deterministic: {deterministic}")
