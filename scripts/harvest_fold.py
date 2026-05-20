#!/usr/bin/env python3
"""K-fold cross-prediction harvester for the merge classifier (ADR-0002).

For one ``--fold k``:
  1. Read the folds manifest written by ``prepare_folds.py``.
  2. Train a fresh PointUNet on the **other** four folds' training scenes,
     validating on the dataset's original val split (so val is never leaked
     into harvested pairs).
  3. Run inference on the held-out fold's scenes:
     embeddings → clusterer → cluster graph → pairwise features +
     dominant-GT-face labels → JSONL.
  4. Persist the fold's checkpoint and pair JSONL under ``--output-dir``.

Designed to be run once per fold (k = 0..n_folds-1), in any order, possibly
in parallel on a multi-GPU box. Each run is independent and idempotent
(except that re-running overwrites the same fold's artefacts).

Heavy lift — requires a working CUDA / PyG environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a fold model and harvest cluster-pair training data from its predictions."
    )
    p.add_argument("--fold", type=int, required=True, help="fold index (0..n_folds-1)")
    p.add_argument(
        "--folds-manifest",
        default="configs/folds.json",
        help="folds JSON written by scripts/prepare_folds.py",
    )
    p.add_argument("--data-root", default="data/roofNTNU/train_test_split")
    p.add_argument("--output-dir", default="artifacts/merge_classifier")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=None, help="overrides Config.batch_size")
    p.add_argument(
        "--clusterer",
        default="hdbscan",
        choices=["hdbscan", "dbscan", "meanshift"],
    )
    p.add_argument("--adjacency-radius", type=float, default=0.05)
    p.add_argument("--min-purity", type=float, default=0.6)
    p.add_argument(
        "--reuse-checkpoint",
        default=None,
        help="if set, skip training and use this checkpoint (for re-running harvest only)",
    )
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    # Heavy imports (PyG, torch) — guarded so --help is fast.
    import numpy as np
    import torch
    from torch.utils.data import Subset

    from roofseg.device import describe_device, select_device
    from roofseg.models import default_config
    from roofseg.refinement.folds import load_folds
    from roofseg.refinement.training_data import harvest_pairs, write_pairs_jsonl
    from roofseg.seed import set_seed
    from roofseg.training import train_pointunet

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {describe_device(device)}")

    manifest = load_folds(args.folds_manifest)
    if args.fold < 0 or args.fold >= manifest.n_folds:
        print(
            f"--fold {args.fold} out of range; manifest has n_folds={manifest.n_folds}",
            file=sys.stderr,
        )
        return 2

    heldout_scenes = set(manifest.scenes_for_fold(args.fold))
    train_scene_ids = set(manifest.train_scenes_excluding_fold(args.fold))
    print(
        f"Fold {args.fold}: train_scenes={len(train_scene_ids)} heldout_scenes={len(heldout_scenes)}"
    )

    import trainModel as tm

    config = default_config()
    if args.batch_size:
        config.batch_size = args.batch_size

    full_train = tm.LiDARPointCloudDataset(
        base_dir=args.data_root,
        split=manifest.split,
        max_points=config.max_points,
        sampling_method=config.sampling_method,
    )
    val_split = tm.LiDARPointCloudDataset(
        base_dir=args.data_root,
        split="val",
        max_points=config.max_points,
        sampling_method=config.sampling_method,
    )

    def _stem(name: str) -> str:
        return Path(name).stem

    train_indices = [i for i, fname in enumerate(full_train.point_files) if _stem(fname) in train_scene_ids]
    heldout_indices = [i for i, fname in enumerate(full_train.point_files) if _stem(fname) in heldout_scenes]
    if not train_indices or not heldout_indices:
        print("fold has empty train or heldout subset — check folds manifest", file=sys.stderr)
        return 3

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = out_dir / f"fold_{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "model.pth"
    log_path = fold_dir / "train_log.json"
    pairs_path = fold_dir / "pairs.jsonl"

    if args.reuse_checkpoint:
        print(f"Reusing checkpoint: {args.reuse_checkpoint}")
        from roofseg.models import load_pointunet
        model = load_pointunet(args.reuse_checkpoint, config=config, map_location=device).to(device)
    else:
        print(f"Training fold {args.fold} for {args.epochs} epochs ...")
        t0 = time.time()
        train_subset = Subset(full_train, train_indices)
        model, _hist = train_pointunet(
            train_subset,
            val_split,
            config=config,
            epochs=args.epochs,
            save_path=ckpt_path,
            device=device,
            log_path=log_path,
        )
        print(f"Training done in {time.time() - t0:.0f}s; saved {ckpt_path}")

    # Inference + pair extraction over the held-out scenes.
    from roofseg.clustering import cluster_embeddings

    print(f"Harvesting pairs over {len(heldout_indices)} held-out scenes ...")
    model.eval()
    n_pairs_total = 0
    n_pairs_pos = 0
    # Clear any stale pairs file before appending.
    if pairs_path.exists():
        pairs_path.unlink()

    with torch.no_grad():
        for idx in heldout_indices:
            scene_id = _stem(full_train.point_files[idx])
            # Deterministic FPS subsample per scene.
            np.random.seed((args.seed + abs(hash(scene_id))) % (2**31 - 1))
            points_t, gt_labels_t, _ = full_train[idx]
            points = points_t[:, :3].numpy().astype(np.float64)
            gt_labels = gt_labels_t.numpy().astype(np.int64)

            embeddings = model(points_t.unsqueeze(0).to(device))[0].cpu().numpy()
            clustering = cluster_embeddings(embeddings, method=args.clusterer)
            records = harvest_pairs(
                points,
                clustering.labels,
                gt_labels,
                scene_id=scene_id,
                fold=args.fold,
                adjacency_radius=args.adjacency_radius,
                min_purity=args.min_purity,
            )
            n = write_pairs_jsonl(records, pairs_path, append=True)
            n_pairs_total += n
            n_pairs_pos += sum(1 for r in records if r.should_merge)

    summary = {
        "fold": args.fold,
        "n_train_scenes": len(train_indices),
        "n_heldout_scenes": len(heldout_indices),
        "n_pairs_total": n_pairs_total,
        "n_pairs_should_merge": n_pairs_pos,
        "positive_rate": n_pairs_pos / n_pairs_total if n_pairs_total else 0.0,
        "clusterer": args.clusterer,
        "adjacency_radius": args.adjacency_radius,
        "min_purity": args.min_purity,
        "checkpoint": str(ckpt_path) if not args.reuse_checkpoint else args.reuse_checkpoint,
        "pairs_file": str(pairs_path),
    }
    with open(fold_dir / "harvest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
