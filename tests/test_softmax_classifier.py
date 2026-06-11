"""Tests for the closed-set softmax baseline's Hungarian-matching loss.

The encoder requires PyG so it isn't exercised here; the matching logic and
loss function are pure torch + scipy and run locally.
"""

from __future__ import annotations

import torch

from roofseg.baselines.softmax_classifier import (
    HungarianClassificationLoss,
    hungarian_match_labels,
)


def test_hungarian_match_identity_when_logits_align_with_gt():
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ]
    )
    gt = torch.tensor([0, 0, 1, 1, 2, 2])
    out = hungarian_match_labels(logits, gt, num_classes=3)
    assert torch.equal(out, gt)


def test_hungarian_match_remaps_permuted_gt():
    # Logits favor [2,2,0,0,1,1]; GT is [0,0,1,1,2,2] — should remap to match.
    logits = torch.tensor(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    )
    gt = torch.tensor([0, 0, 1, 1, 2, 2])
    out = hungarian_match_labels(logits, gt, num_classes=3)
    expected = torch.tensor([2, 2, 0, 0, 1, 1])
    assert torch.equal(out, expected)


def test_hungarian_match_preserves_padding():
    logits = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    gt = torch.tensor([-1, 0, 1])
    out = hungarian_match_labels(logits, gt, num_classes=3)
    assert out[0].item() == -1
    assert out[1].item() == 1
    assert out[2].item() == 2


def test_hungarian_match_drops_unmatched_when_more_gt_than_classes():
    # 4 GT ids but K=2 — only 2 GT ids can be matched, the rest go to ignore.
    logits = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ]
    )
    gt = torch.tensor([0, 1, 2, 3])
    out = hungarian_match_labels(logits, gt, num_classes=2)
    matched = out[out != -1]
    assert len(matched) == 2
    assert set(matched.tolist()) == {0, 1}


def test_hungarian_classification_loss_improves_with_better_predictions():
    loss_fn = HungarianClassificationLoss(num_classes=3)
    gt = torch.tensor([[0, 0, 1, 1, 2, 2]])
    uniform = torch.zeros((1, 6, 3))
    confident = torch.zeros((1, 6, 3))
    confident[0, 0:2, 0] = 10.0
    confident[0, 2:4, 1] = 10.0
    confident[0, 4:6, 2] = 10.0
    assert loss_fn(confident, gt).item() < loss_fn(uniform, gt).item()


def test_hungarian_classification_loss_near_zero_when_perfect():
    loss_fn = HungarianClassificationLoss(num_classes=3)
    gt = torch.tensor([[0, 0, 1, 1, 2, 2]])
    perfect = torch.zeros((1, 6, 3))
    perfect[0, 0:2, 0] = 50.0
    perfect[0, 2:4, 1] = 50.0
    perfect[0, 4:6, 2] = 50.0
    loss = loss_fn(perfect, gt)
    assert loss.item() < 1e-3


def test_hungarian_classification_loss_handles_padding():
    loss_fn = HungarianClassificationLoss(num_classes=3)
    gt = torch.tensor([[-1, 0, 1, -1]])
    logits = torch.zeros((1, 4, 3))
    logits[0, 1, 0] = 50.0
    logits[0, 2, 1] = 50.0
    # The two real points are perfectly classified; padding should not blow up.
    loss = loss_fn(logits, gt)
    assert loss.item() < 1e-3
