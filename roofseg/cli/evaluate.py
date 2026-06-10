"""Reproducible CLI eval (replaces the per-checkpoint logic in ``testneva*.ipynb``).

Loads a PointUNet checkpoint, runs the canonical inference pipeline over a
data split, and writes a JSON identical in shape to the existing
``evaluation_results/evaluation_summary.json`` plus the new headline metrics
(``number_of_faces_error``, ``over_segmentation_rate``, ``under_segmentation_rate``).

Importing the model triggers PyTorch Geometric, which can be fragile in some
environments. To keep the package importable everywhere, all PyTorch/PyG-touching
imports happen inside :func:`main`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roofseg-evaluate",
        description="Evaluate a PointUNet checkpoint with the canonical inference pipeline.",
    )
    p.add_argument("--checkpoint", required=True, help="path to .pth state_dict")
    p.add_argument(
        "--data-root",
        default="data/roofNTNU/train_test_split",
        help="folder containing points_{split}_n and labels_{split}_n",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--clusterer",
        default="hdbscan",
        choices=["hdbscan", "dbscan", "meanshift"],
    )
    p.add_argument(
        "--no-refinement",
        action="store_true",
        help="disable plane-aware refinement (baseline ablation row)",
    )
    p.add_argument(
        "--no-noise-recovery",
        action="store_true",
        help="disable the noise-point recovery stage",
    )
    p.add_argument("--merge-threshold", type=float, default=0.5)
    p.add_argument("--adjacency-radius", type=float, default=0.05)
    p.add_argument(
        "--scorer-model",
        default=None,
        help="path to a trained LightGBM merge classifier (.txt); defaults to HandTunedScorer",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="force a device; default: best available",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-json",
        default="evaluation_results/evaluation_summary.json",
        help="where to write the per-scene metrics",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N scenes (smoke-test)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _add_repo_root_to_syspath()

    # Heavy imports happen here so `python -m roofseg.cli.evaluate --help` is fast
    # and the package itself doesn't transitively load PyG.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from roofseg.device import describe_device, select_device
    from roofseg.metrics import aggregate, scene_metrics
    from roofseg.seed import set_seed

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {describe_device(device)}")

    # PyG-touching imports: model + clustering + refinement.
    from roofseg.clustering import cluster_embeddings
    from roofseg.inference import PipelineConfig, run_inference
    from roofseg.models import default_config, load_pointunet

    # Dataset comes from the legacy module so eval matches training subsampling.
    import trainModel as tm

    config = default_config()
    model = load_pointunet(args.checkpoint, config=config, map_location=device).to(device)
    model.eval()

    dataset = tm.LiDARPointCloudDataset(
        base_dir=args.data_root,
        split=args.split,
        max_points=config.max_points,
        sampling_method=config.sampling_method,
    )
    if len(dataset) == 0:
        print(f"no scenes found in {args.data_root}/points_{args.split}_n", file=sys.stderr)
        return 2

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    pipeline_config = PipelineConfig(
        clusterer=args.clusterer,
        apply_refinement=not args.no_refinement,
        recover_noise=not args.no_noise_recovery,
        merge_threshold=args.merge_threshold,
        adjacency_radius=args.adjacency_radius,
    )

    scorer = None
    if args.scorer_model:
        from roofseg.refinement.scoring import LightGBMScorer

        scorer = LightGBMScorer(args.scorer_model)
    scorer_name = scorer.name if scorer is not None else "hand_tuned"

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_scenes = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    print(f"Evaluating {n_scenes} scenes from split={args.split} | clusterer={args.clusterer} "
          f"| refinement={'on' if pipeline_config.apply_refinement else 'off'} "
          f"| scorer={scorer_name} "
          f"| noise_recovery={'on' if pipeline_config.recover_noise else 'off'}")

    with torch.no_grad():
        for idx, (points_t, labels_t, _) in enumerate(loader):
            if args.limit is not None and idx >= args.limit:
                break

            scene_name = f"scene_{idx:03d}"
            points_t = points_t.to(device)
            gt_labels = labels_t.cpu().numpy()[0]

            embeddings = model(points_t)[0].cpu().numpy()
            points_xyz = points_t[0, :, :3].cpu().numpy()

            result = run_inference(points_xyz, embeddings, pipeline_config, scorer=scorer)
            pred_labels = result.final_labels

            valid = gt_labels != -1
            if valid.sum() == 0:
                continue

            metrics = scene_metrics(
                scene_name, gt_labels[valid].astype(np.int64), pred_labels[valid]
            )
            rows.append(metrics.to_dict())
            if idx % 10 == 0:
                print(
                    f"  [{idx + 1}/{n_scenes}] {scene_name}: "
                    f"gt={metrics.gt_instances} pred={metrics.pred_instances} "
                    f"NFE={metrics.number_of_faces_error} ARI={metrics.ARI:.3f}"
                )

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    from roofseg.metrics import SceneMetrics

    summary = aggregate([SceneMetrics(**r) for r in rows])
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote per-scene metrics: {output_path}")
    print(f"Wrote aggregate summary:  {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
