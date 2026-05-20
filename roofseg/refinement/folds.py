"""K-fold split manifest for cross-prediction harvesting (ADR-0002).

The merge classifier needs training data drawn from *realistic*
over-segmentation — i.e. clusterings produced by a model that hasn't seen
those scenes during training. We obtain this via k-fold cross-prediction
on the training split: for each fold, a fresh PointUNet is trained on the
other folds' scenes and used to predict the held-out fold.

This module owns the fold assignment so it stays deterministic across
machines (the heavy training is on the CUDA server, the harvested pairs
come back here for analysis).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class FoldManifest:
    n_folds: int
    seed: int
    split: str  # which dataset split these scenes came from (typically 'train')
    folds: dict[int, list[str]]  # fold_idx -> scene_ids

    def all_scenes(self) -> list[str]:
        return sorted({s for ids in self.folds.values() for s in ids})

    def scenes_for_fold(self, fold_idx: int) -> list[str]:
        if fold_idx not in self.folds:
            raise KeyError(f"fold {fold_idx} not in manifest (have {sorted(self.folds.keys())})")
        return list(self.folds[fold_idx])

    def train_scenes_excluding_fold(self, fold_idx: int) -> list[str]:
        return sorted(
            scene for f, scenes in self.folds.items() for scene in scenes if f != fold_idx
        )

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "split": self.split,
            "folds": {str(k): v for k, v in sorted(self.folds.items())},
        }


def make_folds(
    scene_ids: list[str], *, n_folds: int = 5, seed: int = 42, split: str = "train"
) -> FoldManifest:
    """Deterministic fold assignment.

    Shuffles a copy of ``scene_ids`` with a fixed seed and round-robins
    across folds so fold sizes differ by at most one. Sorting the input
    first guarantees the result is invariant to caller ordering.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    if not scene_ids:
        raise ValueError("scene_ids is empty")

    sorted_ids = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(sorted_ids)

    folds: dict[int, list[str]] = {i: [] for i in range(n_folds)}
    for idx, scene in enumerate(sorted_ids):
        folds[idx % n_folds].append(scene)

    for k in folds:
        folds[k].sort()  # stable on-disk order

    return FoldManifest(n_folds=n_folds, seed=seed, split=split, folds=folds)


def save_folds(manifest: FoldManifest, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_folds(path: str | Path) -> FoldManifest:
    with open(path) as f:
        raw = json.load(f)
    return FoldManifest(
        n_folds=int(raw["n_folds"]),
        seed=int(raw["seed"]),
        split=raw.get("split", "train"),
        folds={int(k): list(v) for k, v in raw["folds"].items()},
    )
