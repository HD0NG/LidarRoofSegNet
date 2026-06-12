#!/usr/bin/env python3
"""Dump per-point predictions for qualitative figures (server-side).

For each selected scene, runs the three pipeline configurations that appear in
the ablation (raw / hand-tuned / LightGBM@0.6) and saves an ``.npz`` containing
the points, GT labels, and each configuration's predicted labels. The heavy
(PyG / model) part lives here; ``scripts/plot_qualitative.py`` renders locally
from the saved ``.npz`` without touching the model.

Default scenes were chosen from the per-scene metrics to span the story:
  17 — complex roof (GT=6), raw over-segments to 11, LightGBM recovers 6.
  37 — moderate (GT=4), raw 9 -> LightGBM 4.
  75 — flat roof (GT=1), raw splits into 5 spurious faces, LightGBM collapses to 1.
   2 — simple (GT=2), shows refinement does not damage easy scenes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_SCENES = [17, 37, 75, 2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dump per-point predictions for qualitative figures.")
    p.add_argument("--checkpoint", default="roof_segmentation_dgcnn_best.pth")
    p.add_argument("--scorer-model", default="artifacts/merge_classifier/model.txt")
    p.add_argument("--merge-threshold", type=float, default=0.6)
    p.add_argument("--data-root", default="data/roofNTNU/train_test_split")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--scenes", type=int, nargs="+", default=DEFAULT_SCENES)
    p.add_argument("--output-dir", default="evaluation_results/qualitative")
    p.add_argument("--clusterer", default="hdbscan", choices=["hdbscan", "dbscan", "meanshift"])
    p.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import numpy as np
    import torch

    from roofseg.device import describe_device, select_device
    from roofseg.inference import PipelineConfig, run_inference
    from roofseg.models import default_config, load_pointunet
    from roofseg.refinement.scoring import LightGBMScorer
    from roofseg.seed import set_seed
    import trainModel as tm

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {describe_device(device)}")

    config = default_config()
    model = load_pointunet(args.checkpoint, config=config, map_location=device).to(device)
    model.eval()
    scorer = LightGBMScorer(args.scorer_model)

    dataset = tm.LiDARPointCloudDataset(
        base_dir=args.data_root, split=args.split,
        max_points=config.max_points, sampling_method=config.sampling_method,
    )

    cfg_raw = PipelineConfig(clusterer=args.clusterer, apply_refinement=False)
    cfg_ht = PipelineConfig(clusterer=args.clusterer, apply_refinement=True, merge_threshold=0.5)
    cfg_lgb = PipelineConfig(clusterer=args.clusterer, apply_refinement=True, merge_threshold=args.merge_threshold)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in args.scenes:
            if idx >= len(dataset):
                print(f"  scene {idx} out of range (n={len(dataset)}) — skipping")
                continue
            points_t, labels_t, _ = dataset[idx]
            points_t = points_t.unsqueeze(0).to(device)
            gt = labels_t.cpu().numpy().astype(np.int64)
            xyz = points_t[0, :, :3].cpu().numpy()
            emb = model(points_t)[0].cpu().numpy()

            raw = run_inference(xyz, emb, cfg_raw).final_labels
            ht = run_inference(xyz, emb, cfg_ht).final_labels
            lgb = run_inference(xyz, emb, cfg_lgb, scorer=scorer).final_labels

            out_path = out_dir / f"scene_{idx:03d}.npz"
            np.savez_compressed(
                out_path, points=xyz, gt=gt, raw=raw, handtuned=ht, lgb=lgb,
            )
            def _n(a):
                return int(np.unique(a[a != -1]).size)
            print(f"  scene {idx:03d}: gt={_n(gt)} raw={_n(raw)} ht={_n(ht)} lgb={_n(lgb)} -> {out_path}")

    print(f"\nWrote {len(args.scenes)} scene dumps to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
