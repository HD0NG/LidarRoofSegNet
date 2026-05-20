"""Sanity tests for the plane-aware refinement skeleton.

Two synthetic scenes drive the basic correctness checks:
  - Two co-planar squares that share a boundary → must merge.
  - Two perpendicular squares (a roof ridge) → must NOT merge.
"""

from __future__ import annotations

import numpy as np

from roofseg.refinement import (
    HandTunedScorer,
    build_cluster_graph,
    fit_plane,
    pairwise_features,
    refine,
)


def _make_grid(x0: float, x1: float, y0: float, y1: float, z_fn, step: float = 0.05) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    Z = z_fn(X, Y)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def test_fit_plane_flat():
    pts = _make_grid(0, 1, 0, 1, z_fn=lambda x, y: np.zeros_like(x))
    plane = fit_plane(pts)
    # Normal points along +Z (we orient towards +Z).
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1e-6)
    assert plane.residual < 1e-9
    assert plane.slope < 1e-3  # arccos near 1 is float-noisy; ~0.06° tolerance


def test_two_coplanar_clusters_merge():
    pts_a = _make_grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _make_grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(pts_a), dtype=np.int64), np.ones(len(pts_b), dtype=np.int64)]
    )

    scorer = HandTunedScorer(
        normal_angle_tol=0.1, joint_residual_tol=0.01, boundary_distance_tol=0.1, offset_tol=0.05
    )
    result = refine(points, labels, scorer=scorer, merge_threshold=0.5, adjacency_radius=0.1)
    assert result.n_input_clusters == 2
    assert result.n_output_clusters == 1
    assert len(np.unique(result.labels)) == 1


def test_perpendicular_clusters_do_not_merge():
    # Ridge: cluster A is the +x-facing pitch, B is the -x-facing pitch.
    pts_a = _make_grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: x)  # 45° slope, +x
    pts_b = _make_grid(1.0, 2.0, 0.0, 1.0, z_fn=lambda x, y: 2.0 - x)  # 45° slope, -x
    points = np.concatenate([pts_a, pts_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(pts_a), dtype=np.int64), np.ones(len(pts_b), dtype=np.int64)]
    )

    scorer = HandTunedScorer(
        normal_angle_tol=0.1, joint_residual_tol=0.01, boundary_distance_tol=0.1, offset_tol=0.05
    )
    result = refine(points, labels, scorer=scorer, merge_threshold=0.5, adjacency_radius=0.1)
    assert result.n_input_clusters == 2
    assert result.n_output_clusters == 2


def test_cluster_graph_adjacency_is_symmetric():
    pts_a = _make_grid(0.0, 0.5, 0.0, 0.5, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _make_grid(0.55, 1.0, 0.0, 0.5, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(pts_a), dtype=np.int64), np.ones(len(pts_b), dtype=np.int64)]
    )

    graph = build_cluster_graph(points, labels, adjacency_radius=0.1)
    assert 1 in graph.neighbours(0)
    assert 0 in graph.neighbours(1)


def test_pairwise_features_basic_shape():
    pts_a = _make_grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _make_grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    labels = np.concatenate(
        [np.zeros(len(pts_a), dtype=np.int64), np.ones(len(pts_b), dtype=np.int64)]
    )
    graph = build_cluster_graph(points, labels, adjacency_radius=0.1)
    feats = pairwise_features(graph.nodes[0], graph.nodes[1], points)
    vec = feats.to_vector()
    assert vec.shape == (11,)
    # Co-planar squares → near-zero normal angle and joint residual.
    assert feats.normal_angle < 1e-3
    assert feats.joint_residual < 1e-6


def test_noise_label_excluded_from_graph():
    pts_a = _make_grid(0.0, 0.5, 0.0, 0.5, z_fn=lambda x, y: np.zeros_like(x))
    noise = np.random.RandomState(0).rand(20, 3)
    points = np.concatenate([pts_a, noise], axis=0)
    labels = np.concatenate(
        [np.zeros(len(pts_a), dtype=np.int64), np.full(len(noise), -1, dtype=np.int64)]
    )
    graph = build_cluster_graph(points, labels, adjacency_radius=0.1)
    assert set(graph.nodes.keys()) == {0}
