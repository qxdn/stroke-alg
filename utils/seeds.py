import random
import torch
import numpy as np
import sklearn

def set_seed(seed: int = 114514, deterministic: bool = False):
    if not seed:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    sklearn.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)
