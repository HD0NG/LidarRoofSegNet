"""Deterministic seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed all stdlib/NumPy/PyTorch RNGs.

    `deterministic_cudnn` trades a small amount of speed for reproducible
    cuDNN kernels; turn it off for benchmarking.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
