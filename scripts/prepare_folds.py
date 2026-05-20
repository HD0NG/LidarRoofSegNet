#!/usr/bin/env python3
"""Write a deterministic k-fold manifest for cross-prediction harvesting.

Runs locally — no CUDA needed. Reads scene ids from the training split's
``points_train_n`` folder, shuffles with a fixed seed, partitions into
``--n-folds`` folds, writes ``--output`` JSON.

Example:
    python scripts/prepare_folds.py --output configs/folds.json --n-folds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create the k-fold manifest.")
    p.add_argument("--data-root", default="data/roofNTNU/train_test_split")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="configs/folds.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from roofseg.data import list_scenes
    from roofseg.refinement.folds import make_folds, save_folds

    scene_ids = list_scenes(args.data_root, args.split)
    if len(scene_ids) < args.n_folds:
        print(
            f"only {len(scene_ids)} scenes found in split={args.split}; "
            f"need at least n_folds={args.n_folds}",
            file=sys.stderr,
        )
        return 2

    manifest = make_folds(scene_ids, n_folds=args.n_folds, seed=args.seed, split=args.split)
    save_folds(manifest, args.output)

    print(f"Wrote {args.output}")
    print(f"  split={args.split} | n_scenes={len(scene_ids)} | n_folds={args.n_folds} | seed={args.seed}")
    for k in sorted(manifest.folds):
        print(f"  fold {k}: {len(manifest.folds[k])} scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
