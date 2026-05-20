"""Lazy loaders for the existing PointUNet checkpoints.

Importing ``trainModel`` triggers a PyTorch Geometric load, which can be
fragile (torch_scatter ABI mismatches). We defer that import until the
caller actually needs a model.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import torch


def _import_train_model() -> Any:
    """Import ``trainModel`` from the repo root, regardless of cwd."""
    if "trainModel" in sys.modules:
        return sys.modules["trainModel"]

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module("trainModel")


def default_config() -> Any:
    """Return the legacy ``Config`` class instance from trainModel."""
    tm = _import_train_model()
    return tm.Config()


def build_pointunet(config: Any | None = None) -> torch.nn.Module:
    """Construct an un-trained PointUNet using the legacy Config defaults."""
    tm = _import_train_model()
    if config is None:
        config = tm.Config()
    return tm.PointUNet(config)


def load_pointunet(
    checkpoint_path: str | os.PathLike,
    config: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build a PointUNet and load weights from ``checkpoint_path``.

    The checkpoint must be a plain ``state_dict`` (what ``train_pipeline`` saves).
    """
    model = build_pointunet(config)
    state = torch.load(str(checkpoint_path), map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model
