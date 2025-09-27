import random

import numpy as np
import torch


def seed_everything(seed: int = 0):
    cuda_available = torch.cuda.is_available()
    print(f"Setting all random seeds to {seed}, cuda_available={cuda_available}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('medium')
