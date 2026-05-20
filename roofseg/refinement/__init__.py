"""Plane-aware refinement: cluster-graph reasoning that merges over-segmented
clusters into roof faces.

The headline contribution of the paper (ADR-0001). Each cluster from the
unsupervised clusterer is a graph node carrying a fitted plane (normal,
residual, size, ...). Adjacent clusters are scored by a **learned merge
classifier** (ADR-0002) and merged pairwise-greedily.

Encoder-agnostic by design: features live in :mod:`roofseg.refinement.features`
and are computed purely from XYZ, never from the embedding space.
"""

from roofseg.refinement.cluster_graph import (
    ClusterGraph,
    ClusterNode,
    build_cluster_graph,
    fit_plane,
)
from roofseg.refinement.features import PairFeatures, pairwise_features
from roofseg.refinement.folds import (
    FoldManifest,
    load_folds,
    make_folds,
    save_folds,
)
from roofseg.refinement.greedy_merge import greedy_merge
from roofseg.refinement.pipeline import refine
from roofseg.refinement.scoring import (
    HandTunedScorer,
    LightGBMScorer,
    MergeScorer,
)
from roofseg.refinement.training_data import (
    PairRecord,
    dominant_label,
    harvest_pairs,
    load_pairs_jsonl,
    load_pairs_many,
    write_pairs_jsonl,
)

__all__ = [
    "ClusterGraph",
    "ClusterNode",
    "FoldManifest",
    "HandTunedScorer",
    "LightGBMScorer",
    "MergeScorer",
    "PairFeatures",
    "PairRecord",
    "build_cluster_graph",
    "dominant_label",
    "fit_plane",
    "greedy_merge",
    "harvest_pairs",
    "load_folds",
    "load_pairs_jsonl",
    "load_pairs_many",
    "make_folds",
    "pairwise_features",
    "refine",
    "save_folds",
    "write_pairs_jsonl",
]
