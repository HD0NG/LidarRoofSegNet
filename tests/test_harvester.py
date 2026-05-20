"""Tests for the pair-record harvester."""

from __future__ import annotations

import json

import numpy as np

from roofseg.refinement.training_data import (
    dominant_label,
    feature_columns,
    harvest_pairs,
    load_pairs_jsonl,
    write_pairs_jsonl,
)


def _grid(x0, x1, y0, y1, z_fn, step=0.05):
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel(), z_fn(X, Y).ravel()], axis=1)


def test_dominant_label_pure_clusters():
    cluster = np.array([0, 0, 0, 1, 1, 1])
    gt = np.array([10, 10, 10, 20, 20, 20])
    out = dominant_label(cluster, gt)
    assert out[0] == (10, 1.0)
    assert out[1] == (20, 1.0)


def test_dominant_label_ignores_gt_noise():
    cluster = np.array([0, 0, 0, 0])
    gt = np.array([10, 10, -1, -1])
    out = dominant_label(cluster, gt)
    assert out[0][0] == 10
    assert out[0][1] == 1.0  # purity computed on non-noise only


def test_dominant_label_mixed_cluster_purity():
    cluster = np.array([0, 0, 0, 0, 0])
    gt = np.array([10, 10, 10, 20, 20])  # 3/5 = 0.6 purity for 10
    out = dominant_label(cluster, gt)
    assert out[0][0] == 10
    assert out[0][1] == 0.6


def test_dominant_label_skips_all_noise_cluster():
    cluster = np.array([0, 0, 1, 1])
    gt = np.array([-1, -1, 20, 20])
    out = dominant_label(cluster, gt)
    assert 0 not in out  # all-noise cluster dropped
    assert 1 in out


def test_harvest_pairs_same_gt_face_labelled_positive():
    # Two adjacent halves of the same flat GT face → same dominant GT id → merge.
    pts_a = _grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    cluster = np.concatenate([np.zeros(len(pts_a)), np.ones(len(pts_b))]).astype(np.int64)
    gt = np.full(len(points), 7, dtype=np.int64)  # one GT face spanning both clusters

    records = harvest_pairs(
        points, cluster, gt, scene_id="s", fold=0, adjacency_radius=0.1, min_purity=0.5
    )
    assert len(records) == 1
    assert records[0].should_merge is True
    assert records[0].dom_a == 7 and records[0].dom_b == 7


def test_harvest_pairs_different_gt_face_labelled_negative():
    pts_a = _grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    cluster = np.concatenate([np.zeros(len(pts_a)), np.ones(len(pts_b))]).astype(np.int64)
    gt = np.concatenate([np.full(len(pts_a), 1), np.full(len(pts_b), 2)]).astype(np.int64)

    records = harvest_pairs(
        points, cluster, gt, scene_id="s", fold=0, adjacency_radius=0.1, min_purity=0.5
    )
    assert len(records) == 1
    assert records[0].should_merge is False


def test_harvest_pairs_drops_low_purity_pair():
    # Cluster 0 has 50/50 GT 1/2 → purity 0.5; threshold 0.7 should drop the pair.
    pts_a = _grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    cluster = np.concatenate([np.zeros(len(pts_a)), np.ones(len(pts_b))]).astype(np.int64)
    n_a = len(pts_a)
    gt = np.concatenate(
        [np.array([1] * (n_a // 2) + [2] * (n_a - n_a // 2)), np.full(len(pts_b), 1)]
    ).astype(np.int64)

    high = harvest_pairs(
        points, cluster, gt, scene_id="s", fold=0, adjacency_radius=0.1, min_purity=0.7
    )
    low = harvest_pairs(
        points, cluster, gt, scene_id="s", fold=0, adjacency_radius=0.1, min_purity=0.4
    )
    assert len(high) == 0
    assert len(low) == 1


def test_pair_record_jsonl_round_trip(tmp_path):
    pts_a = _grid(0.0, 1.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    pts_b = _grid(1.05, 2.0, 0.0, 1.0, z_fn=lambda x, y: np.zeros_like(x))
    points = np.concatenate([pts_a, pts_b], axis=0)
    cluster = np.concatenate([np.zeros(len(pts_a)), np.ones(len(pts_b))]).astype(np.int64)
    gt = np.full(len(points), 7, dtype=np.int64)

    records = harvest_pairs(
        points, cluster, gt, scene_id="abc", fold=2, adjacency_radius=0.1, min_purity=0.5
    )
    path = tmp_path / "pairs.jsonl"
    n = write_pairs_jsonl(records, path, append=False)
    assert n == len(records)

    df = load_pairs_jsonl(path)
    assert len(df) == n
    for col in ["scene_id", "fold", "should_merge", "purity_a", "purity_b"]:
        assert col in df.columns
    for col in feature_columns():
        assert col in df.columns
    assert df.iloc[0]["scene_id"] == "abc"
    assert df.iloc[0]["fold"] == 2
    assert bool(df.iloc[0]["should_merge"]) is True
