"""Cluster-graph construction.

A :class:`ClusterNode` represents one cluster from the unsupervised
clusterer with its fitted plane and basic geometry summaries. The
:class:`ClusterGraph` carries the nodes plus an adjacency derived from
spatial proximity (KDTree, configurable radius).

Noise (cluster id ``-1``) is excluded from the graph; recovery of noise
points is a separate downstream step (see :mod:`roofseg.inference`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class FittedPlane:
    """Plane fit to a set of points using SVD (least-squares total residual).

    The normal is unit length and oriented towards +Z (so slope is in [0, π/2]).
    """

    normal: np.ndarray  # (3,) unit
    centroid: np.ndarray  # (3,)
    offset: float  # plane: normal · (x - centroid) = 0, equivalently normal · x = offset
    residual: float  # RMS distance of fit points to the plane (metres in input units)

    @property
    def slope(self) -> float:
        """Angle between plane normal and +Z (radians); flat roof → 0."""
        nz = float(np.clip(abs(self.normal[2]), 0.0, 1.0))
        return float(np.arccos(nz))

    @property
    def aspect(self) -> float:
        """Azimuth of the plane normal projected onto XY (radians, atan2(ny, nx))."""
        return float(np.arctan2(self.normal[1], self.normal[0]))


def fit_plane(points: np.ndarray) -> FittedPlane:
    """SVD plane fit. Requires at least 3 points; returns a degenerate plane otherwise."""
    if points.shape[0] < 3:
        centroid = points.mean(axis=0) if points.shape[0] else np.zeros(3)
        return FittedPlane(
            normal=np.array([0.0, 0.0, 1.0]),
            centroid=centroid,
            offset=float(centroid[2]),
            residual=float("inf"),
        )

    centroid = points.mean(axis=0)
    centered = points - centroid
    # Smallest right-singular vector of centered is the plane normal. The
    # corresponding singular value gives the RMS residual without a second matmul.
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    residual = float(s[-1] / np.sqrt(max(points.shape[0], 1)))
    offset = float(normal @ centroid)
    return FittedPlane(normal=normal, centroid=centroid, offset=offset, residual=residual)


@dataclass
class ClusterNode:
    cluster_id: int
    point_indices: np.ndarray  # indices into the scene's full point array
    plane: FittedPlane

    @property
    def size(self) -> int:
        return int(self.point_indices.size)


@dataclass
class ClusterGraph:
    """Cluster nodes + symmetric adjacency.

    ``adjacency`` is a ``dict[int, set[int]]`` keyed by cluster_id. The
    adjacency is built once at construction and updated in place by the
    merge loop as clusters are combined.
    """

    nodes: dict[int, ClusterNode]
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    def edges(self) -> Iterable[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        for a, neighbours in self.adjacency.items():
            for b in neighbours:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                yield key

    def neighbours(self, cluster_id: int) -> set[int]:
        return self.adjacency.get(cluster_id, set())


def _adjacency_via_kdtree(
    points: np.ndarray,
    labels: np.ndarray,
    cluster_ids: np.ndarray,
    radius: float,
) -> dict[int, set[int]]:
    """Two clusters are adjacent if any point in cluster A has a point in
    cluster B within ``radius``.

    Implementation: build one KDTree over all non-noise points, query
    ``query_ball_point`` per point, then add an edge for any neighbour with
    a different label.
    """
    valid_mask = labels != -1
    valid_points = points[valid_mask]
    valid_labels = labels[valid_mask]
    if valid_points.shape[0] == 0:
        return {int(c): set() for c in cluster_ids}

    tree = cKDTree(valid_points)
    adj: dict[int, set[int]] = {int(c): set() for c in cluster_ids}

    neighbour_lists = tree.query_ball_point(valid_points, r=radius)
    for i, neigh in enumerate(neighbour_lists):
        li = int(valid_labels[i])
        for j in neigh:
            lj = int(valid_labels[j])
            if lj == li:
                continue
            adj[li].add(lj)
            adj[lj].add(li)
    return adj


def build_cluster_graph(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    adjacency_radius: float = 0.05,
) -> ClusterGraph:
    """Construct a cluster graph from points + per-point cluster labels.

    Args:
        points: ``(N, 3)`` XYZ.
        labels: ``(N,)`` cluster ids (``-1`` = noise; excluded from the graph).
        adjacency_radius: two clusters are adjacent if any point in one is within
            this distance of any point in the other. Should be on the order of
            the point spacing of the scene. The default (0.05) suits the
            roofNTNU normalised scale; rescale for raw-metric scenes.
    """
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must be (N, 3+); got {points.shape}")
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"labels length {labels.shape[0]} != points length {points.shape[0]}"
        )

    xyz = points[:, :3].astype(np.float64, copy=False)
    cluster_ids = np.unique(labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    nodes: dict[int, ClusterNode] = {}
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        plane = fit_plane(xyz[idx])
        nodes[int(cid)] = ClusterNode(cluster_id=int(cid), point_indices=idx, plane=plane)

    adjacency = _adjacency_via_kdtree(xyz, labels, cluster_ids, radius=adjacency_radius)
    return ClusterGraph(nodes=nodes, adjacency=adjacency)
