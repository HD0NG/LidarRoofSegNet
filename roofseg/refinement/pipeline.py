"""High-level plane-aware refinement entrypoint.

Combines :func:`build_cluster_graph` + :func:`greedy_merge` + relabelling
into a single call. Use this from the canonical inference pipeline in
:mod:`roofseg.inference`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roofseg.refinement.cluster_graph import ClusterGraph, build_cluster_graph
from roofseg.refinement.greedy_merge import (
    MergeTrace,
    graph_to_labels,
    greedy_merge,
    relabel_compact,
)
from roofseg.refinement.scoring import MergeScorer, default_scorer


@dataclass
class RefinementResult:
    labels: np.ndarray
    graph: ClusterGraph
    trace: MergeTrace
    n_input_clusters: int
    n_output_clusters: int


def refine(
    points: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    scorer: MergeScorer | None = None,
    merge_threshold: float = 0.5,
    adjacency_radius: float = 0.05,
    compact_labels: bool = True,
    record_trace: bool = False,
) -> RefinementResult:
    """Run plane-aware refinement on top of an over-segmented cluster labelling.

    Args:
        points: ``(N, 3+)`` scene points; only XYZ is used.
        cluster_labels: ``(N,)`` cluster ids from the unsupervised clusterer
            (``-1`` = noise; preserved through to the output).
        scorer: any :class:`MergeScorer`. Defaults to :class:`HandTunedScorer`
            until the LightGBM model is trained.
        merge_threshold: minimum merge score to accept.
        adjacency_radius: clusters within this distance are considered adjacent.
        compact_labels: if True, remap output ids to ``0..K-1``.
        record_trace: if True, record every merge event in the result.
    """
    if scorer is None:
        scorer = default_scorer()

    graph = build_cluster_graph(points, cluster_labels, adjacency_radius=adjacency_radius)
    n_in = len(graph.nodes)

    trace = MergeTrace() if record_trace else MergeTrace()  # always allocate; cheap
    greedy_merge(
        graph,
        points,
        scorer,
        merge_threshold=merge_threshold,
        trace=trace if record_trace else None,
    )

    refined = graph_to_labels(graph, n_points=points.shape[0], original_labels=cluster_labels)
    if compact_labels:
        refined = relabel_compact(refined)

    return RefinementResult(
        labels=refined,
        graph=graph,
        trace=trace,
        n_input_clusters=n_in,
        n_output_clusters=len(graph.nodes),
    )
