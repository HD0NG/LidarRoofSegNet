"""Pair-record harvester for the learned merge classifier (ADR-0002).

For each held-out scene under k-fold cross-prediction:

1. Take the model's over-segmented cluster labels.
2. Build the cluster graph (same adjacency the inference-time refinement uses).
3. For every adjacent cluster pair (a, b):
   * Compute the 11-d pairwise geometric features.
   * Compute the dominant GT roof-face id for each cluster (ignoring noise).
   * If either cluster's dominant-face purity is below ``min_purity``, drop
     the pair — its label would be unreliable.
   * Label ``should_merge = (dom_a == dom_b)``.

The output is a JSON-lines file: one record per pair. Downstream the
LightGBM trainer (:mod:`roofseg.refinement.classifier_training`) loads
the JSONL into a pandas DataFrame and splits by scene to avoid leakage.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from roofseg.refinement.cluster_graph import build_cluster_graph
from roofseg.refinement.features import FEATURE_ORDER, PairFeatures, pairwise_features


@dataclass
class PairRecord:
    scene_id: str
    fold: int
    cluster_a: int
    cluster_b: int
    should_merge: bool
    dom_a: int  # dominant GT face id for cluster a (-1 if undecidable)
    dom_b: int
    purity_a: float  # fraction of cluster_a points belonging to dom_a
    purity_b: float
    size_a: int
    size_b: int
    features: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float | int | bool | str]:
        out: dict[str, float | int | bool | str] = {
            "scene_id": self.scene_id,
            "fold": self.fold,
            "cluster_a": self.cluster_a,
            "cluster_b": self.cluster_b,
            "should_merge": bool(self.should_merge),
            "dom_a": self.dom_a,
            "dom_b": self.dom_b,
            "purity_a": float(self.purity_a),
            "purity_b": float(self.purity_b),
            "size_a": self.size_a,
            "size_b": self.size_b,
        }
        for k in FEATURE_ORDER:
            out[f"f_{k}"] = float(self.features[k])
        return out


def dominant_label(
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    ignore_gt: tuple[int, ...] = (-1,),
) -> dict[int, tuple[int, float]]:
    """For each non-noise cluster, return ``(dominant_gt_id, purity)``.

    Purity is the fraction of the cluster's *non-ignored* points that share
    the dominant GT label. Clusters whose points all belong to ``ignore_gt``
    are omitted from the result.
    """
    out: dict[int, tuple[int, float]] = {}
    ignore_set = set(ignore_gt)
    cluster_ids = np.unique(cluster_labels)
    cluster_ids = cluster_ids[cluster_ids != -1]

    for cid in cluster_ids:
        mask = cluster_labels == cid
        cluster_gt = gt_labels[mask]
        kept = cluster_gt[~np.isin(cluster_gt, list(ignore_set))]
        if kept.size == 0:
            continue
        vals, counts = np.unique(kept, return_counts=True)
        idx = int(np.argmax(counts))
        out[int(cid)] = (int(vals[idx]), float(counts[idx] / kept.size))
    return out


def harvest_pairs(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    *,
    scene_id: str,
    fold: int,
    adjacency_radius: float,
    min_purity: float = 0.6,
) -> list[PairRecord]:
    """Extract one pair record per adjacent cluster pair.

    Pairs where either cluster lacks a confident dominant GT face (purity
    below ``min_purity``) are dropped — they would be label-noise for the
    classifier.
    """
    if points.shape[0] != cluster_labels.shape[0]:
        raise ValueError(
            f"length mismatch points={points.shape[0]} cluster_labels={cluster_labels.shape[0]}"
        )
    if points.shape[0] != gt_labels.shape[0]:
        raise ValueError(
            f"length mismatch points={points.shape[0]} gt_labels={gt_labels.shape[0]}"
        )

    graph = build_cluster_graph(points, cluster_labels, adjacency_radius=adjacency_radius)
    dom = dominant_label(cluster_labels, gt_labels)

    records: list[PairRecord] = []
    for a, b in graph.edges():
        if a not in dom or b not in dom:
            continue
        dom_a, pur_a = dom[a]
        dom_b, pur_b = dom[b]
        if pur_a < min_purity or pur_b < min_purity:
            continue
        feats = pairwise_features(graph.nodes[a], graph.nodes[b], points)
        records.append(
            PairRecord(
                scene_id=scene_id,
                fold=fold,
                cluster_a=int(a),
                cluster_b=int(b),
                should_merge=(dom_a == dom_b),
                dom_a=int(dom_a),
                dom_b=int(dom_b),
                purity_a=float(pur_a),
                purity_b=float(pur_b),
                size_a=int(graph.nodes[a].size),
                size_b=int(graph.nodes[b].size),
                features=feats.to_dict(),
            )
        )
    return records


def write_pairs_jsonl(
    records: Iterable[PairRecord], path: str | Path, *, append: bool = False
) -> int:
    """Write pair records to JSON-lines. Returns the number of records written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(p, mode) as f:
        for rec in records:
            f.write(json.dumps(rec.to_flat_dict()) + "\n")
            n += 1
    return n


def load_pairs_jsonl(path: str | Path):
    """Load pair records into a pandas DataFrame. Imports pandas lazily."""
    import pandas as pd

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_pairs_many(paths: Iterable[str | Path]):
    """Concatenate multiple pair JSONL files into one DataFrame."""
    import pandas as pd

    frames = [load_pairs_jsonl(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def feature_columns() -> list[str]:
    return [f"f_{k}" for k in FEATURE_ORDER]
