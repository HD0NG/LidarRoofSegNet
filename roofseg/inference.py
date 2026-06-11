"""Canonical inference pipeline (CONTEXT.md):

    embeddings → clusterer → plane-aware refinement → noise recovery → labels

Each stage is pluggable. ``knn_smoothing`` and ``ransac_refinement`` from the
legacy :mod:`post_processing` are **disabled by default** — they're kept as
ablation rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from roofseg.clustering import ClusteringResult, cluster_embeddings
from roofseg.refinement.pipeline import RefinementResult, refine
from roofseg.refinement.scoring import MergeScorer


@dataclass
class PipelineConfig:
    clusterer: Literal["hdbscan", "dbscan", "meanshift"] = "hdbscan"
    clusterer_overrides: dict[str, Any] = field(default_factory=dict)

    apply_refinement: bool = True
    merge_threshold: float = 0.5
    adjacency_radius: float = 0.05

    recover_noise: bool = True
    noise_recovery_k: int = 10
    noise_recovery_normal_threshold: float = 0.8

    # Legacy ablation toggles — off by default, never on in the headline config.
    apply_knn_smoothing: bool = False
    knn_smoothing_k: int = 10
    apply_ransac_refinement: bool = False
    ransac_residual_threshold: float = 0.1


@dataclass
class PipelineResult:
    points: np.ndarray
    raw_cluster_labels: np.ndarray
    refined_labels: np.ndarray | None
    final_labels: np.ndarray
    clustering: ClusteringResult
    refinement: RefinementResult | None


def run_inference(
    points: np.ndarray,
    embeddings: np.ndarray | None,
    config: PipelineConfig,
    *,
    scorer: MergeScorer | None = None,
    cluster_labels: np.ndarray | None = None,
) -> PipelineResult:
    """Run the canonical pipeline.

    When ``cluster_labels`` is provided (baseline path), they are used directly
    and ``embeddings`` is ignored. Otherwise ``embeddings`` is clustered with
    the configured clusterer; ``points`` and ``embeddings`` must align on axis 0.
    """
    if cluster_labels is not None:
        if cluster_labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"points/cluster_labels length mismatch: "
                f"{points.shape[0]} vs {cluster_labels.shape[0]}"
            )
        clustering = ClusteringResult(
            labels=cluster_labels.astype(np.int64).copy(),
            method="precomputed",
        )
    else:
        if embeddings is None:
            raise ValueError("provide either embeddings or cluster_labels")
        if points.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"points/embeddings length mismatch: {points.shape[0]} vs {embeddings.shape[0]}"
            )
        clustering = cluster_embeddings(
            embeddings, method=config.clusterer, **config.clusterer_overrides
        )
    labels = clustering.labels.copy()

    refinement: RefinementResult | None = None
    refined_labels: np.ndarray | None = None
    if config.apply_refinement:
        refinement = refine(
            points,
            labels,
            scorer=scorer,
            merge_threshold=config.merge_threshold,
            adjacency_radius=config.adjacency_radius,
        )
        refined_labels = refinement.labels
        labels = refined_labels

    if config.recover_noise:
        # Lazy import to keep refinement module PyG-free.
        from post_processing import RoofPostProcessor

        labels = RoofPostProcessor.recover_noise_points(
            points,
            labels,
            k=config.noise_recovery_k,
            normal_threshold=config.noise_recovery_normal_threshold,
        )

    if config.apply_knn_smoothing:
        from post_processing import RoofPostProcessor

        labels = RoofPostProcessor.knn_smoothing(points, labels, k=config.knn_smoothing_k)

    if config.apply_ransac_refinement:
        from post_processing import RoofPostProcessor

        labels = RoofPostProcessor.ransac_refinement(
            points, labels, residual_threshold=config.ransac_residual_threshold
        )

    return PipelineResult(
        points=points,
        raw_cluster_labels=clustering.labels,
        refined_labels=refined_labels,
        final_labels=labels,
        clustering=clustering,
        refinement=refinement,
    )
