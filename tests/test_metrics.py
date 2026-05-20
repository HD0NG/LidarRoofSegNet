"""Sanity tests for the metrics module."""

from __future__ import annotations

import math

import numpy as np

from roofseg.metrics import (
    aggregate,
    iou_matrix,
    number_of_faces_error,
    over_under_segmentation_rates,
    scene_metrics,
)


def test_iou_matrix_perfect_match():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1])
    iou, gt_ids, pred_ids, gt_sizes, pred_sizes = iou_matrix(gt, pred)
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(np.diag(iou), [1.0, 1.0])
    np.testing.assert_array_equal(gt_sizes, [3, 3])
    np.testing.assert_array_equal(pred_sizes, [3, 3])


def test_iou_matrix_ignores_padding():
    gt = np.array([0, 0, -1, 1, 1])
    pred = np.array([0, 0, 0, 1, 1])
    iou, gt_ids, pred_ids, *_ = iou_matrix(gt, pred)
    # GT noise excluded → only gt ids 0 and 1
    assert set(gt_ids.tolist()) == {0, 1}
    assert set(pred_ids.tolist()) == {0, 1}


def test_number_of_faces_error_overseg():
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 2, 2, 3])  # 4 predicted, 2 truth → |4-2| = 2
    assert number_of_faces_error(gt, pred) == 2


def test_number_of_faces_error_exact():
    gt = np.array([0, 0, 1, 1])
    pred = np.array([5, 5, 7, 7])  # ids don't matter, only counts
    assert number_of_faces_error(gt, pred) == 0


def test_over_segmentation_rate_detects_split():
    # GT face 0 is split into two predicted clusters (pred 0 and pred 1),
    # each covering half its points → both above IoU 0.4 against gt face 0.
    gt = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    iou, *_ = iou_matrix(gt, pred)
    over, under = over_under_segmentation_rates(iou, iou_match_threshold=0.4)
    # gt face 0 matched by pred 0 AND pred 1 above threshold (both IoU = 2/4 = 0.5)
    assert over > 0.0
    assert under == 0.0


def test_under_segmentation_rate_detects_merge():
    # Two GT faces collapsed into one predicted cluster.
    gt = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    iou, *_ = iou_matrix(gt, pred)
    over, under = over_under_segmentation_rates(iou, iou_match_threshold=0.4)
    # pred 0 matches both gt 0 (IoU 0.5) and gt 1 (IoU 0.5) ≥ 0.4
    assert under > 0.0


def test_scene_metrics_single_gt_class_yields_nan_ari():
    # CONTEXT.md: ARI is undefined when ground truth has only one class.
    gt = np.zeros(10, dtype=np.int64)
    pred = np.zeros(10, dtype=np.int64)
    m = scene_metrics("scene_x", gt, pred)
    assert math.isnan(m.ARI)
    assert math.isnan(m.NMI)
    assert m.gt_instances == 1


def test_scene_metrics_basic_fields_present():
    gt = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    m = scene_metrics("scene_x", gt, pred)
    d = m.to_dict()
    for key in [
        "ARI",
        "NMI",
        "mIoU",
        "mCov",
        "mWCov",
        "mRec",
        "mPrec",
        "mWPrec",
        "gt_instances",
        "pred_instances",
        "number_of_faces_error",
        "over_segmentation_rate",
        "under_segmentation_rate",
        "complexity",
    ]:
        assert key in d
    assert d["gt_instances"] == 3
    assert d["pred_instances"] == 3
    assert d["number_of_faces_error"] == 0
    assert d["complexity"] == "moderate"


def test_aggregate_nan_safe():
    rows = [
        scene_metrics("a", np.array([0, 0, 0]), np.array([0, 0, 0])),  # nan ARI
        scene_metrics("b", np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])),
    ]
    summary = aggregate(rows)
    # mean_ARI should ignore the NaN row and still produce a finite value.
    assert math.isfinite(summary["mean_ARI"])
    assert summary["n_scenes"] == 2.0
