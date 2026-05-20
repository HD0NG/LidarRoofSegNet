"""Pairwise geometric features for the merge classifier.

Encoder-agnostic by design (ADR-0002 + CONTEXT.md): features are purely
geometric — no embedding-space contribution — so the refinement transfers
to any over-segmenting clusterer, not only DGCNN + HDBSCAN.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree

from roofseg.refinement.cluster_graph import ClusterNode, fit_plane


FEATURE_ORDER: tuple[str, ...] = (
    "normal_angle",
    "residual_a",
    "residual_b",
    "joint_residual",
    "boundary_distance",
    "size_a",
    "size_b",
    "size_ratio",
    "slope_diff",
    "aspect_diff",
    "offset_diff",
)


@dataclass
class PairFeatures:
    normal_angle: float  # radians, in [0, π/2] after sign-folding
    residual_a: float
    residual_b: float
    joint_residual: float  # RMS distance of A∪B points to the plane fit on A∪B
    boundary_distance: float  # min point-to-point distance between A and B
    size_a: int
    size_b: int
    size_ratio: float  # min(size_a, size_b) / max(size_a, size_b) in [0, 1]
    slope_diff: float  # |slope_a - slope_b|, radians
    aspect_diff: float  # wrap-around aware, radians in [0, π]
    offset_diff: float  # |normal_a · centroid_a - normal_b · centroid_b|

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        d = self.to_dict()
        return np.array([d[k] for k in FEATURE_ORDER], dtype=np.float64)


def _wrap_angle(angle: float) -> float:
    """Wrap a radian angle into ``[0, π]``."""
    a = abs(angle) % (2 * np.pi)
    return float(min(a, 2 * np.pi - a))


def pairwise_features(
    node_a: ClusterNode,
    node_b: ClusterNode,
    points: np.ndarray,
) -> PairFeatures:
    """Compute the pairwise feature vector for one candidate merge edge.

    Args:
        node_a, node_b: cluster nodes carrying fitted planes and point indices.
        points: ``(N, 3+)`` scene points; only ``[:, :3]`` is used.
    """
    xyz = points[:, :3].astype(np.float64, copy=False)
    pts_a = xyz[node_a.point_indices]
    pts_b = xyz[node_b.point_indices]

    # 1. Normal angle (sign-folded so we never penalise antiparallel normals).
    cos_sim = float(np.clip(abs(np.dot(node_a.plane.normal, node_b.plane.normal)), 0.0, 1.0))
    normal_angle = float(np.arccos(cos_sim))

    # 2. Per-cluster residuals (already cached on the fitted planes).
    res_a = float(node_a.plane.residual)
    res_b = float(node_b.plane.residual)

    # 3. Joint-fit residual: how well does a single plane explain A ∪ B?
    joint_pts = np.concatenate([pts_a, pts_b], axis=0)
    joint_plane = fit_plane(joint_pts)
    joint_residual = float(joint_plane.residual)

    # 4. Boundary distance: minimum point-to-point distance between A and B.
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        boundary_distance = float("inf")
    else:
        tree = cKDTree(pts_b)
        dists, _ = tree.query(pts_a, k=1)
        boundary_distance = float(dists.min())

    # 5. Sizes + ratio.
    size_a = int(pts_a.shape[0])
    size_b = int(pts_b.shape[0])
    denom = max(size_a, size_b)
    size_ratio = float(min(size_a, size_b) / denom) if denom > 0 else 0.0

    # 6. Slope and aspect differences.
    slope_diff = float(abs(node_a.plane.slope - node_b.plane.slope))
    aspect_diff = _wrap_angle(node_a.plane.aspect - node_b.plane.aspect)

    # 7. Plane-offset difference (signed offset along each plane's own normal).
    offset_diff = float(abs(node_a.plane.offset - node_b.plane.offset))

    return PairFeatures(
        normal_angle=normal_angle,
        residual_a=res_a,
        residual_b=res_b,
        joint_residual=joint_residual,
        boundary_distance=boundary_distance,
        size_a=size_a,
        size_b=size_b,
        size_ratio=size_ratio,
        slope_diff=slope_diff,
        aspect_diff=aspect_diff,
        offset_diff=offset_diff,
    )
