"""Tests for the normal-based region growing baseline (ADR-0001)."""

from __future__ import annotations

from collections import Counter

import numpy as np

from roofseg.baselines.region_growing import (
    RegionGrowingConfig,
    region_growing_segment,
)


def _two_planes(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_each = 1000

    plane1 = np.column_stack(
        [
            rng.uniform(0.0, 5.0, n_each),
            rng.uniform(0.0, 5.0, n_each),
            rng.normal(0.0, 0.005, n_each),
        ]
    )
    x2 = rng.uniform(20.0, 25.0, n_each)  # 15-unit gap so KNN can't bridge
    y2 = rng.uniform(0.0, 5.0, n_each)
    z2 = (x2 - 20.0) * np.tan(np.radians(30.0)) + rng.normal(0.0, 0.005, n_each)
    plane2 = np.column_stack([x2, y2, z2])

    points = np.vstack([plane1, plane2])
    truth = np.concatenate([np.zeros(n_each, dtype=np.int64), np.ones(n_each, dtype=np.int64)])
    return points, truth


def _dominant_label(labels: np.ndarray) -> int:
    valid = labels[labels != -1]
    if valid.size == 0:
        return -1
    return int(Counter(valid.tolist()).most_common(1)[0][0])


def test_region_growing_separates_two_planes():
    points, truth = _two_planes()
    labels = region_growing_segment(points)

    found = np.unique(labels[labels != -1])
    assert len(found) == 2, f"expected exactly 2 clusters, got {found.tolist()}"

    dom1 = _dominant_label(labels[truth == 0])
    dom2 = _dominant_label(labels[truth == 1])
    assert dom1 != dom2 and dom1 != -1 and dom2 != -1


def test_region_growing_returns_all_unassigned_when_too_few_points():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(10, 3))
    labels = region_growing_segment(points)
    assert labels.shape == (10,)
    assert np.all(labels == -1)


def test_region_growing_respects_min_cluster_size():
    # Build the same two-plane scene but require clusters of size 5000 — neither
    # plane qualifies, so every point should be unassigned.
    points, _ = _two_planes()
    cfg = RegionGrowingConfig(min_cluster_size=5000)
    labels = region_growing_segment(points, cfg)
    assert np.all(labels == -1)


def test_region_growing_merges_coplanar_points():
    # A single flat plane should produce exactly one cluster (no spurious splits).
    rng = np.random.default_rng(2)
    n = 1500
    points = np.column_stack(
        [
            rng.uniform(0.0, 5.0, n),
            rng.uniform(0.0, 5.0, n),
            rng.normal(0.0, 0.005, n),
        ]
    )
    labels = region_growing_segment(points)
    found = np.unique(labels[labels != -1])
    assert len(found) == 1
