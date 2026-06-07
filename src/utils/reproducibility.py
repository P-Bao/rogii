"""
Seed and reproducibility utilities.
Call set_seed() at the top of every training script.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_env_info() -> dict:
    """Return environment info for reproducibility logging."""
    info = {
        'python': __import__('sys').version,
        'numpy': np.__version__,
    }
    for lib in ['pandas', 'lightgbm', 'xgboost', 'sklearn', 'torch']:
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            info[lib] = 'not installed'
    return info
