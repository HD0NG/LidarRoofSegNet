"""Device selection: portable across CUDA, MPS, and CPU."""

from __future__ import annotations

import torch


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the best available torch device.

    Args:
        prefer: optional explicit choice ('cuda', 'mps', 'cpu'). If the
            requested device is unavailable, falls back to the best alternative.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"
