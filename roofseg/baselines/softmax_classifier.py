"""Closed-set softmax baseline (Baseline 0 per ADR-0001).

DGCNN encoder + K-way per-point classification head, trained with
Hungarian-matching cross-entropy so the loss is invariant to the arbitrary
instance-label permutations across scenes. Serves as the "is the embedding
paradigm worth it" reference; plane-aware refinement does not apply to it
(the head emits class IDs directly).

The encoder import is deferred to ``SoftmaxClassifier.__init__`` so this
module imports cleanly on environments without PyG (e.g. local Mac); only
constructing the model triggers the PyG load.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_match_labels(
    logits: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Permute per-point GT labels to align with the predicted classes.

    Each GT instance id is matched to the predicted class with which it
    shares the most log-probability mass via Hungarian assignment. The
    matching is computed on detached logits so it does not contribute to
    the gradient.

    When the scene has more distinct GT ids than ``num_classes``, the
    surplus ids cannot be matched and their points are set to
    ``ignore_index`` so they're dropped by the downstream cross-entropy.

    Args:
        logits: ``(N, K)`` per-point class logits.
        gt_labels: ``(N,)`` integer GT instance ids; ``ignore_index`` = padding.
        num_classes: ``K``.
        ignore_index: label used for padding and for unmatched GT ids.

    Returns:
        ``(N,)`` integer labels: ``[0..K-1]`` for matched points,
        ``ignore_index`` for padding and unmatched GT ids.
    """
    valid = gt_labels != ignore_index
    if not valid.any():
        return gt_labels.clone()

    unique_gt = torch.unique(gt_labels[valid]).tolist()
    log_probs = F.log_softmax(logits, dim=-1).detach()

    n_gt = len(unique_gt)
    cost = torch.zeros((n_gt, num_classes), dtype=torch.float64)
    for i, g in enumerate(unique_gt):
        mask = gt_labels == g
        cost[i] = -log_probs[mask].sum(dim=0).cpu().double()

    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

    out = gt_labels.clone()
    out[valid] = ignore_index  # drop unmatched valid points by default
    for ri, ci in zip(row_ind, col_ind):
        g = unique_gt[ri]
        out[gt_labels == g] = int(ci)
    return out


class HungarianClassificationLoss(nn.Module):
    """Hungarian-matching cross-entropy for closed-set instance prediction.

    For each scene in the batch, find the optimal assignment between GT
    instance ids and predicted classes, permute the targets accordingly, then
    apply standard cross-entropy. Gradients flow through cross-entropy; the
    matching step uses detached logits.
    """

    def __init__(self, num_classes: int, *, ignore_index: int = -1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        B, N, K = logits.shape
        if K != self.num_classes:
            raise ValueError(
                f"logits last dim {K} != num_classes {self.num_classes}"
            )
        permuted = torch.empty_like(gt_labels)
        for b in range(B):
            permuted[b] = hungarian_match_labels(
                logits[b], gt_labels[b], K, ignore_index=self.ignore_index
            )
        return F.cross_entropy(
            logits.reshape(-1, K),
            permuted.reshape(-1),
            ignore_index=self.ignore_index,
        )


class SoftmaxClassifier(nn.Module):
    """DGCNN encoder + K-way per-point classification head."""

    def __init__(self, config, num_classes: int) -> None:
        super().__init__()
        # Lazy import: trainModel.py loads PyG eagerly, which segfaults on
        # environments without a working torch_geometric.
        from trainModel import DGCNNEncoder

        self.encoder = DGCNNEncoder(
            input_dim=config.input_dim,
            k=config.k_neighbors,
            emb_dim=config.emb_dim,
        )
        self.classifier_head = nn.Sequential(
            nn.Conv1d(config.emb_dim * 2, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, N, C)`` -> per-point logits ``(B, N, K)``."""
        x = x.permute(0, 2, 1)  # (B, C, N)
        features = self.encoder(x)  # (B, 2*emb_dim, N)
        logits = self.classifier_head(features)  # (B, K, N)
        return logits.permute(0, 2, 1)  # (B, N, K)
