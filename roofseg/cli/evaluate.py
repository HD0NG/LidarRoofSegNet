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
    p.add_argument(
        "--checkpoint",
        default=None,
        help="path to .pth state_dict; required unless --baseline is set",
    )
    p.add_argument(
        "--baseline",
        default=None,
        choices=["region_growing", "softmax"],
        help="run a baseline instead of the embedding pipeline; "
        "region_growing skips any model, softmax loads --softmax-checkpoint",
    )
    p.add_argument(
        "--softmax-checkpoint",
        default=None,
        help="state_dict for the closed-set softmax baseline (required when --baseline softmax)",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=8,
        help="K for the softmax baseline; ignored otherwise",
    )
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
    if args.baseline is None and args.checkpoint is None:
        print(
            "--checkpoint is required unless --baseline is set", file=sys.stderr
        )
        return 2
    if args.baseline == "softmax" and args.softmax_checkpoint is None:
        print(
            "--softmax-checkpoint is required when --baseline softmax",
            file=sys.stderr,
        )
        return 2
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

    # Refinement / inference imports (PyG-free).
    from roofseg.inference import PipelineConfig, run_inference
    from roofseg.models import default_config  # safe — PyG load is lazy inside load_pointunet

    # Dataset comes from the legacy module so eval matches training subsampling.
    import trainModel as tm

    config = default_config()
    model = None
    softmax_model = None
    if args.baseline is None:
        # PyG-touching import deferred until we actually need the model.
        from roofseg.models import load_pointunet

        model = load_pointunet(args.checkpoint, config=config, map_location=device).to(device)
        model.eval()
    elif args.baseline == "softmax":
        from roofseg.baselines.softmax_classifier import SoftmaxClassifier

        softmax_model = SoftmaxClassifier(
            config=config, num_classes=args.num_classes
        ).to(device)
        state = torch.load(args.softmax_checkpoint, map_location=device)
        softmax_model.load_state_dict(state)
        softmax_model.eval()
    else:  # region_growing
        from roofseg.baselines import region_growing_segment  # noqa: F401

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
    clusterer_name = args.baseline if args.baseline else args.clusterer
    print(f"Evaluating {n_scenes} scenes from split={args.split} | clusterer={clusterer_name} "
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

            points_xyz = points_t[0, :, :3].cpu().numpy()
            if args.baseline == "region_growing":
                cluster_labels = region_growing_segment(points_xyz)
                result = run_inference(
                    points_xyz, None, pipeline_config,
                    scorer=scorer, cluster_labels=cluster_labels,
                )
            elif args.baseline == "softmax":
                logits = softmax_model(points_t)[0]  # (N, K)
                cluster_labels = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                result = run_inference(
                    points_xyz, None, pipeline_config,
                    scorer=scorer, cluster_labels=cluster_labels,
                )
            else:
                embeddings = model(points_t)[0].cpu().numpy()
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
