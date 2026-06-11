"""Normal-based region growing baseline (Rabbani 2006 style).

A pre-deep-learning baseline for roof-face instance segmentation. Per-point
normals are estimated by PCA on the k-nearest-neighbour patch, curvature is
taken as the smallest eigenvalue's share of the total, and regions grow from
low-curvature seed points to neighbours whose normals align within tolerance.

Encoder-free: takes raw points -> integer cluster labels. Plug-compatible with
the canonical pipeline as an alternative clusterer; ADR-0001 requires every
baseline to be evaluated both with and without plane-aware refinement on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class RegionGrowingConfig:
    knn: int = 30
    normal_angle_tol_rad: float = 0.20  # ~11.5 degrees
    curvature_threshold: float = 0.04
    min_cluster_size: int = 50


def _estimate_normals_and_curvature(
    points: np.ndarray, knn: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(normals (N,3), curvature (N,), neighbour_indices (N, knn))``."""
    n = points.shape[0]
    nn = NearestNeighbors(n_neighbors=knn).fit(points)
    _, indices = nn.kneighbors(points)

    normals = np.zeros((n, 3), dtype=np.float64)
    curvature = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbr = points[indices[i]]
        centred = nbr - nbr.mean(axis=0, keepdims=True)
        cov = centred.T @ centred / max(nbr.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
        normals[i] = eigvecs[:, 0]
        total = eigvals.sum()
        curvature[i] = float(eigvals[0] / total) if total > 0 else 0.0
    return normals, curvature, indices


def region_growing_segment(
    points: np.ndarray,
    config: RegionGrowingConfig | None = None,
) -> np.ndarray:
    """Cluster points into roof-face candidates via normal-based region growing.

    Args:
        points: ``(N, 3+)`` array; only XYZ is used.
        config: optional override; defaults to :class:`RegionGrowingConfig`.

    Returns:
        ``(N,)`` int array of cluster ids in ``[0..K-1]``, ``-1`` for points
        that ended up in a cluster smaller than ``min_cluster_size`` or were
        never reached.
    """
    cfg = config or RegionGrowingConfig()
    n = points.shape[0]
    if n < cfg.knn:
        return np.full(n, -1, dtype=np.int64)

    xyz = points[:, :3].astype(np.float64)
    normals, curvature, neighbours = _estimate_normals_and_curvature(xyz, cfg.knn)

    labels = np.full(n, -1, dtype=np.int64)
    seed_order = np.argsort(curvature)  # flattest points first
    cos_tol = float(np.cos(cfg.normal_angle_tol_rad))
    next_label = 0

    for seed in seed_order:
        if labels[seed] != -1:
            continue
        cluster: list[int] = []
        queue = [int(seed)]
        labels[seed] = next_label
        while queue:
            p = queue.pop()
            cluster.append(p)
            for q in neighbours[p]:
                q = int(q)
                if labels[q] != -1:
                    continue
                # Use abs to be antipodal-agnostic; PCA normals have no orientation.
                if abs(float(normals[p] @ normals[q])) < cos_tol:
                    continue
                labels[q] = next_label
                # Only re-seed from low-curvature points to keep the region planar.
                if curvature[q] < cfg.curvature_threshold:
                    queue.append(q)
        if len(cluster) < cfg.min_cluster_size:
            for idx in cluster:
                labels[idx] = -1
        else:
            next_label += 1

    return labels
