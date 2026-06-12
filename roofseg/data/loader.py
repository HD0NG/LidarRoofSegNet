"""Load individual roofNTNU scenes from the on-disk `points_*_n` / `labels_*_n` layout.

The training pipeline goes through :class:`LiDARPointCloudDataset` (from
``trainModel.py``), which subsamples / pads to a fixed point count. Eval
needs the *full* scene at native resolution, so this module reads the
.txt files directly without touching PyTorch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SceneSample:
    """One labelled point cloud — one row in the evaluation summary."""

    scene_id: str
    points: np.ndarray  # (N, 3) float64
    instance_labels: np.ndarray  # (N,) int64; -1 means padding/no-label


def _load_txt(path: str | os.PathLike, n_cols: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip().replace(",", " ")
            if not stripped:
                continue
            try:
                values = np.fromstring(stripped, sep=" ", dtype=np.float64)
            except ValueError:
                continue
            if values.size == n_cols:
                rows.append(values)
    if not rows:
        return np.zeros((0, n_cols), dtype=np.float64)
    return np.stack(rows, axis=0)


def list_scenes(data_root: str | os.PathLike, split: str) -> list[str]:
    """Return sorted scene ids present in ``points_{split}_n``."""
    points_dir = Path(data_root) / f"points_{split}_n"
    if not points_dir.exists():
        raise FileNotFoundError(f"points folder not found: {points_dir}")
    return sorted(p.stem for p in points_dir.glob("*.txt"))


def load_scene(
    data_root: str | os.PathLike,
    split: str,
    scene_id: str,
    *,
    drop_padding: bool = True,
) -> SceneSample:
    """Load one scene as raw points + instance labels.

    Args:
        data_root: e.g. ``data/roofNTNU/train_test_split``.
        split: ``train`` / ``val`` / ``test``.
        scene_id: filename stem; ``points_{split}_n/{scene_id}.txt`` must exist.
        drop_padding: filter rows with label ``-1`` (padded / no-label).
    """
    root = Path(data_root)
    points_path = root / f"points_{split}_n" / f"{scene_id}.txt"
    labels_path = root / f"labels_{split}_n" / f"{scene_id}.txt"

    points = _load_txt(points_path, n_cols=3)
    raw_labels = _load_txt(labels_path, n_cols=1).reshape(-1)

    if raw_labels.size != points.shape[0]:
        # Fall back to truncation/padding rather than crashing; warn loudly.
        n = min(raw_labels.size, points.shape[0])
        points = points[:n]
        raw_labels = raw_labels[:n]

    labels = raw_labels.astype(np.int64)
    if drop_padding:
        keep = labels != -1
        points = points[keep]
        labels = labels[keep]

    return SceneSample(scene_id=scene_id, points=points, instance_labels=labels)


def iter_scenes(
    data_root: str | os.PathLike,
    split: str,
    *,
    drop_padding: bool = True,
) -> Iterator[SceneSample]:
    """Yield every scene in a split in sorted order."""
    for scene_id in list_scenes(data_root, split):
        yield load_scene(data_root, split, scene_id, drop_padding=drop_padding)
