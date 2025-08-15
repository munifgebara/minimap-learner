# Sets global seeds for full reproducibility.
import random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    """Seed python, numpy and torch (including dataloader workers)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init(worker_id: int) -> None:
    """Initialize each DataLoader worker with a deterministic seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
