"""Instance-segmentation metrics for roof-face evaluation.

Ports the metrics from ``testneva*.ipynb`` into a single reusable module, and
adds the **refinement-targeted** metrics the JUFO L2 paper needs to surface:
``number_of_faces_error``, ``over_segmentation_rate``, ``under_segmentation_rate``.

Padding labels (``-1``) MUST be filtered upstream; see :func:`scene_metrics` for
the canonical entrypoint that does this filtering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def _unique_non_padding(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    return ids[ids != -1]


def iou_matrix(
    gt_labels: np.ndarray, pred_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-pair IoU matrix between GT and pred instances.

    Returns ``(iou, gt_ids, pred_ids, gt_sizes, pred_sizes)``. Noise (-1) is
    excluded from both axes.
    """
    gt_ids = _unique_non_padding(gt_labels)
    pred_ids = _unique_non_padding(pred_labels)

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    gt_sizes = np.zeros(n_gt, dtype=np.int64)
    pred_sizes = np.zeros(n_pred, dtype=np.int64)

    if n_gt == 0 or n_pred == 0:
        return iou, gt_ids, pred_ids, gt_sizes, pred_sizes

    gt_masks = [gt_labels == g for g in gt_ids]
    pred_masks = [pred_labels == p for p in pred_ids]
    gt_sizes = np.array([m.sum() for m in gt_masks], dtype=np.int64)
    pred_sizes = np.array([m.sum() for m in pred_masks], dtype=np.int64)

    for i, gm in enumerate(gt_masks):
        for j, pm in enumerate(pred_masks):
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            union = np.logical_or(gm, pm).sum()
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou, gt_ids, pred_ids, gt_sizes, pred_sizes


# --------------------------------------------------------------------------
# IoU-based metrics (ported from testneva_3.ipynb)
# --------------------------------------------------------------------------


def coverage_metrics(
    iou: np.ndarray, gt_sizes: np.ndarray, pred_sizes: np.ndarray, iou_threshold: float = 0.5
) -> dict[str, float]:
    """mCov / mWCov / mRec / mPrec / mWPrec.

    Returns NaN-safe zeros when either side is empty (matches the legacy
    notebook behaviour).
    """
    if iou.size == 0 or iou.shape[0] == 0 or iou.shape[1] == 0:
        return dict(mCov=0.0, mWCov=0.0, mRec=0.0, mPrec=0.0, mWPrec=0.0)

    max_iou_per_gt = iou.max(axis=1)
    max_iou_per_pred = iou.max(axis=0)

    mCov = float(max_iou_per_gt.mean())
    gt_total = gt_sizes.sum()
    mWCov = float((max_iou_per_gt * gt_sizes).sum() / gt_total) if gt_total else 0.0
    mRec = float((max_iou_per_gt >= iou_threshold).sum() / len(max_iou_per_gt))
    mPrec = float((max_iou_per_pred >= iou_threshold).sum() / len(max_iou_per_pred))
    pred_total = pred_sizes.sum()
    mWPrec = (
        float(((max_iou_per_pred >= iou_threshold) * pred_sizes).sum() / pred_total)
        if pred_total
        else 0.0
    )
    return dict(mCov=mCov, mWCov=mWCov, mRec=mRec, mPrec=mPrec, mWPrec=mWPrec)


def instance_mean_iou(iou: np.ndarray) -> float:
    """mIoU via Hungarian linear assignment on the IoU matrix."""
    if iou.size == 0 or min(iou.shape) == 0:
        return 0.0
    row_ind, col_ind = linear_sum_assignment(-iou)
    return float(iou[row_ind, col_ind].mean())


# --------------------------------------------------------------------------
# Refinement-targeted metrics (new, ADR-0001)
# --------------------------------------------------------------------------


def number_of_faces_error(gt_labels: np.ndarray, pred_labels: np.ndarray) -> int:
    """|pred_instances - gt_instances|. The paper's headline 'did we count it right?' metric."""
    return int(abs(len(_unique_non_padding(pred_labels)) - len(_unique_non_padding(gt_labels))))


def over_under_segmentation_rates(
    iou: np.ndarray,
    gt_sizes: np.ndarray,
    pred_sizes: np.ndarray,
    *,
    coverage_threshold: float = 0.2,
) -> tuple[float, float]:
    """Coverage-based over-/under-segmentation rates.

    - ``over_segmentation_rate``: fraction of GT faces split across ≥2
      predicted clusters that each cover at least ``coverage_threshold`` of
      the GT face's points (the face was fragmented into multiple non-trivial
      pieces).
    - ``under_segmentation_rate``: fraction of predicted clusters that swallow
      ≥2 GT faces, where each swallowed GT contributes at least
      ``coverage_threshold`` of the prediction's points.

    Earlier versions thresholded IoU directly, which missed realistic
    asymmetric over-segs (a 70%/30% split has neither fragment clearing
    IoU 0.5). Raw intersections are recovered from IoU and sizes via
    ``I = IoU · (G + P) / (1 + IoU)``.
    """
    if iou.size == 0 or min(iou.shape) == 0:
        return 0.0, 0.0

    G = gt_sizes[:, None].astype(np.float64)
    P = pred_sizes[None, :].astype(np.float64)
    intersection = iou * (G + P) / (1.0 + iou)

    gt_coverage = intersection / G
    pred_coverage = intersection / P

    over = (gt_coverage >= coverage_threshold).sum(axis=1) >= 2
    under = (pred_coverage >= coverage_threshold).sum(axis=0) >= 2
    return float(over.mean()), float(under.mean())


# --------------------------------------------------------------------------
# Per-scene rollup
# --------------------------------------------------------------------------


@dataclass
class SceneMetrics:
    scene: str
    ARI: float
    NMI: float
    mIoU: float
    mCov: float
    mWCov: float
    mRec: float
    mPrec: float
    mWPrec: float
    gt_instances: int
    pred_instances: int
    number_of_faces_error: int
    over_segmentation_rate: float
    under_segmentation_rate: float
    complexity: str

    def to_dict(self) -> dict:
        return asdict(self)


def complexity_bucket(num_gt_instances: int) -> str:
    """Same buckets the legacy eval used: simple ≤ 2, moderate ≤ 5, else complex."""
    if num_gt_instances <= 2:
        return "simple"
    if num_gt_instances <= 5:
        return "moderate"
    return "complex"


def scene_metrics(
    scene_id: str,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> SceneMetrics:
    """Compute every per-scene metric on a single scene.

    Caller is responsible for filtering padding (label ``-1``) **before**
    calling — both arrays are assumed to be over the same N points.
    """
    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"shape mismatch: gt={gt_labels.shape}, pred={pred_labels.shape}"
        )

    gt_ids = _unique_non_padding(gt_labels)
    n_gt = len(gt_ids)

    # CONTEXT.md flagged: ARI is undefined when only one GT class.
    if n_gt <= 1:
        ari = float("nan")
        nmi = float("nan")
    else:
        ari = float(adjusted_rand_score(gt_labels, pred_labels))
        nmi = float(normalized_mutual_info_score(gt_labels, pred_labels))

    iou, _, _, gt_sizes, pred_sizes = iou_matrix(gt_labels, pred_labels)
    cov = coverage_metrics(iou, gt_sizes, pred_sizes, iou_threshold=iou_threshold)
    miou = instance_mean_iou(iou)
    n_pred = len(_unique_non_padding(pred_labels))
    nfe = number_of_faces_error(gt_labels, pred_labels)
    over, under = over_under_segmentation_rates(iou, gt_sizes, pred_sizes)

    return SceneMetrics(
        scene=scene_id,
        ARI=ari,
        NMI=nmi,
        mIoU=miou,
        mCov=cov["mCov"],
        mWCov=cov["mWCov"],
        mRec=cov["mRec"],
        mPrec=cov["mPrec"],
        mWPrec=cov["mWPrec"],
        gt_instances=n_gt,
        pred_instances=n_pred,
        number_of_faces_error=nfe,
        over_segmentation_rate=over,
        under_segmentation_rate=under,
        complexity=complexity_bucket(n_gt),
    )


def aggregate(metrics: Iterable[SceneMetrics]) -> dict[str, float]:
    """Means across scenes; NaN-aware (e.g. ARI is NaN when gt_instances=1)."""
    rows = [m.to_dict() for m in metrics]
    if not rows:
        return {}
    numeric_keys = [
        "ARI",
        "NMI",
        "mIoU",
        "mCov",
        "mWCov",
        "mRec",
        "mPrec",
        "mWPrec",
        "number_of_faces_error",
        "over_segmentation_rate",
        "under_segmentation_rate",
    ]
    out: dict[str, float] = {}
    for k in numeric_keys:
        values = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"mean_{k}"] = float(np.nanmean(values))
    out["n_scenes"] = float(len(rows))
    return out
