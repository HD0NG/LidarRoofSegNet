"""Pre-deep-learning baselines required by ADR-0001.

Each baseline produces per-point integer instance labels (``-1`` = unassigned)
in the same shape as :mod:`roofseg.clustering` outputs, so the canonical
pipeline's refinement and noise-recovery stages can run on top.
"""

from roofseg.baselines.region_growing import (
    RegionGrowingConfig,
    region_growing_segment,
)
from roofseg.baselines.softmax_classifier import (
    HungarianClassificationLoss,
    SoftmaxClassifier,
    hungarian_match_labels,
)

__all__ = [
    "RegionGrowingConfig",
    "region_growing_segment",
    "SoftmaxClassifier",
    "HungarianClassificationLoss",
    "hungarian_match_labels",
]
