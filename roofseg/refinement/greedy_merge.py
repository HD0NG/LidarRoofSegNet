"""Pairwise-greedy merge loop.

Score every adjacency edge, take the highest-scoring edge above the merge
threshold, merge those two clusters, recompute the plane and the neighbouring
edges of the merged cluster, and repeat until no edge clears the threshold.

This is the core inference-time algorithm of the plane-aware refinement
(ADR-0001 + ADR-0002 + CONTEXT.md). The merge classifier — passed in as a
:class:`MergeScorer` — supplies the per-edge probability.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from roofseg.refinement.cluster_graph import (
    ClusterGraph,
    ClusterNode,
    fit_plane,
)
from roofseg.refinement.features import PairFeatures, pairwise_features
from roofseg.refinement.scoring import MergeScorer


@dataclass
class MergeStep:
    """One merge event, for tracing/debugging."""

    surviving: int  # cluster id that absorbed the other
    absorbed: int
    score: float
    features: PairFeatures


@dataclass
class MergeTrace:
    steps: list[MergeStep] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)


def _merge_two_nodes(graph: ClusterGraph, surviving: int, absorbed: int, points: np.ndarray) -> None:
    """In-place merge: ``absorbed`` is removed; ``surviving`` grows."""
    a = graph.nodes[surviving]
    b = graph.nodes[absorbed]
    new_indices = np.concatenate([a.point_indices, b.point_indices])
    new_plane = fit_plane(points[new_indices])
    graph.nodes[surviving] = ClusterNode(
        cluster_id=surviving, point_indices=new_indices, plane=new_plane
    )

    # Rewire adjacency: everyone who pointed at ``absorbed`` now points at ``surviving``.
    for neigh in list(graph.adjacency.get(absorbed, set())):
        if neigh == surviving:
            continue
        graph.adjacency[neigh].discard(absorbed)
        graph.adjacency[neigh].add(surviving)
        graph.adjacency[surviving].add(neigh)
    graph.adjacency[surviving].discard(absorbed)
    graph.adjacency.pop(absorbed, None)
    graph.nodes.pop(absorbed, None)


def _score_edge(
    graph: ClusterGraph,
    points: np.ndarray,
    scorer: MergeScorer,
    a: int,
    b: int,
) -> tuple[float, PairFeatures]:
    feats = pairwise_features(graph.nodes[a], graph.nodes[b], points)
    return scorer.score(feats), feats


def greedy_merge(
    graph: ClusterGraph,
    points: np.ndarray,
    scorer: MergeScorer,
    *,
    merge_threshold: float = 0.5,
    max_iter: int | None = None,
    trace: MergeTrace | None = None,
) -> ClusterGraph:
    """Run pairwise-greedy merging in place on ``graph``.

    Args:
        graph: cluster graph from :func:`build_cluster_graph`. Mutated in place.
        points: ``(N, 3+)`` scene points; only XYZ is used.
        scorer: any object satisfying :class:`MergeScorer` (the learned
            classifier or the hand-tuned ablation).
        merge_threshold: edges with score below this are never merged.
        max_iter: hard cap on merge events (None = until convergence).
        trace: optional :class:`MergeTrace` for inspection/debugging.

    Returns the same graph object for chaining.
    """
    if not graph.nodes:
        return graph

    xyz = points[:, :3].astype(np.float64, copy=False)

    # Priority queue: (-score, counter, a, b). Counter breaks ties deterministically.
    pq: list[tuple[float, int, int, int]] = []
    counter = 0
    edge_scores: dict[tuple[int, int], float] = {}

    def _push(a: int, b: int) -> None:
        nonlocal counter
        key = (a, b) if a < b else (b, a)
        score, _feats = _score_edge(graph, xyz, scorer, key[0], key[1])
        edge_scores[key] = score
        if score >= merge_threshold:
            heapq.heappush(pq, (-score, counter, key[0], key[1]))
            counter += 1

    # Seed the queue with every initial adjacency edge.
    for a, b in graph.edges():
        _push(a, b)

    iters = 0
    while pq:
        if max_iter is not None and iters >= max_iter:
            break
        neg_score, _ctr, a, b = heapq.heappop(pq)
        key = (a, b) if a < b else (b, a)

        # Stale entry: edge no longer exists or its score has changed.
        if a not in graph.nodes or b not in graph.nodes:
            continue
        if b not in graph.adjacency.get(a, set()):
            continue
        current = edge_scores.get(key)
        if current is None or abs(current - (-neg_score)) > 1e-9:
            continue
        if current < merge_threshold:
            continue

        # Recompute features once more for the trace (cheap; we already have it,
        # but recomputing keeps the code obvious — premature opt would obscure).
        score, feats = _score_edge(graph, xyz, scorer, a, b)
        if score < merge_threshold:
            edge_scores[key] = score
            continue

        # Keep the larger cluster id stable — by convention we keep the
        # smaller id as the "surviving" one so labels stay compact-ish.
        surviving, absorbed = (a, b) if a < b else (b, a)
        if trace is not None:
            trace.steps.append(
                MergeStep(surviving=surviving, absorbed=absorbed, score=score, features=feats)
            )

        _merge_two_nodes(graph, surviving, absorbed, xyz)
        iters += 1

        # Recompute the merged node's edges and push fresh entries.
        # Drop stale per-edge scores involving the absorbed node.
        stale = [k for k in edge_scores if absorbed in k]
        for k in stale:
            edge_scores.pop(k, None)
        for n in list(graph.neighbours(surviving)):
            _push(surviving, n)

    return graph


def graph_to_labels(graph: ClusterGraph, n_points: int, original_labels: np.ndarray) -> np.ndarray:
    """Project the post-merge graph back to per-point labels.

    Points that were noise (``-1``) in the input remain noise (they're not
    in the graph). Use :mod:`post_processing` or a recovery step downstream
    if you want them re-assigned.
    """
    out = np.full(n_points, -1, dtype=np.int64)
    # Pre-fill with original labels so any cluster ids not present in the
    # graph (shouldn't happen, but defensive) are preserved.
    out[:] = original_labels.astype(np.int64, copy=False)
    # Rewrite labels for every surviving node.
    for cid, node in graph.nodes.items():
        out[node.point_indices] = cid
    # Anything that referred to an absorbed id but isn't in any surviving
    # node's index set falls through to ``-1`` only if it wasn't reassigned;
    # the loop above guarantees correctness because point_indices is the union.
    return out


def relabel_compact(labels: np.ndarray) -> np.ndarray:
    """Remap cluster ids to ``0..K-1`` (noise ``-1`` preserved)."""
    out = labels.copy()
    valid = out != -1
    unique = np.unique(out[valid])
    remap = {old: new for new, old in enumerate(unique)}
    for old, new in remap.items():
        out[(out == old) & valid] = new
    return out
