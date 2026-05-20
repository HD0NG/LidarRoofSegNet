"""Unsupervised clustering of the per-point embeddings.

Wraps sklearn's HDBSCAN / DBSCAN / MeanShift behind a single entrypoint so
the inference pipeline can swap clusterers without leaking sklearn-isms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.cluster import DBSCAN, MeanShift

try:
    from sklearn.cluster import HDBSCAN

    HAS_HDBSCAN = True
except ImportError:  # sklearn < 1.3 — should not happen given pyproject pin
    HAS_HDBSCAN = False

ClusterMethod = Literal["hdbscan", "dbscan", "meanshift"]


@dataclass
class ClusteringResult:
    labels: np.ndarray
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        ids = np.unique(self.labels)
        return int((ids != -1).sum())


_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "hdbscan": dict(
        min_cluster_size=15,
        min_samples=5,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.2,
    ),
    "dbscan": dict(eps=0.5, min_samples=10),
    "meanshift": dict(bandwidth=0.6, bin_seeding=True),
}


def cluster_embeddings(
    embeddings: np.ndarray,
    method: ClusterMethod = "hdbscan",
    **overrides: Any,
) -> ClusteringResult:
    """Cluster (N, D) embeddings into instance ids; ``-1`` denotes noise."""
    method = method.lower()  # type: ignore[assignment]
    if method not in _DEFAULT_PARAMS:
        raise ValueError(f"unknown clustering method: {method!r}")

    params = {**_DEFAULT_PARAMS[method], **overrides}

    if method == "hdbscan":
        if not HAS_HDBSCAN:
            raise RuntimeError("HDBSCAN unavailable (requires scikit-learn >= 1.3)")
        clusterer = HDBSCAN(**params)
    elif method == "dbscan":
        clusterer = DBSCAN(**params)
    else:  # meanshift
        clusterer = MeanShift(**params)

    labels = clusterer.fit_predict(embeddings).astype(np.int64)
    return ClusteringResult(labels=labels, method=method, params=params)
