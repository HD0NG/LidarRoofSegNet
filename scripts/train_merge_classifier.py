#!/usr/bin/env python3
"""Train the LightGBM merge classifier on harvested pair JSONL files (ADR-0002).

Inputs are one or more JSONL files written by ``scripts/harvest_fold.py``.
The classifier is split **by scene** to avoid leakage (adjacent pairs in
the same scene share point geometry). Saves a LightGBM ``.txt`` model
file loadable by :class:`roofseg.refinement.scoring.LightGBMScorer`,
plus a metrics JSON.

Runs on CPU; LightGBM training is quick (~seconds to a minute).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the LightGBM merge classifier.")
    p.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="one or more pair JSONL files (e.g. artifacts/merge_classifier/fold_*/pairs.jsonl)",
    )
    p.add_argument("--output-model", default="artifacts/merge_classifier/model.txt")
    p.add_argument("--output-metrics", default="artifacts/merge_classifier/metrics.json")
    p.add_argument("--val-scene-ratio", type=float, default=0.2)
    p.add_argument("--num-boost-round", type=int, default=500)
    p.add_argument("--early-stopping-rounds", type=int, default=30)
    p.add_argument(
        "--params-json",
        default=None,
        help="optional JSON file with LightGBM param overrides (merged onto DEFAULT_LGB_PARAMS)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from roofseg.refinement.classifier_training import (
        train_merge_classifier,
        write_metrics_json,
    )
    from roofseg.refinement.training_data import load_pairs_many

    # Expand globs the shell didn't (most shells do, but be defensive).
    pair_files: list[Path] = []
    for pattern in args.pairs:
        matches = sorted(Path().glob(pattern))
        pair_files.extend(matches if matches else [Path(pattern)])
    pair_files = [p for p in pair_files if p.exists()]
    if not pair_files:
        print(f"no pair files found from patterns {args.pairs}", file=sys.stderr)
        return 2
    print(f"Loading {len(pair_files)} pair file(s): {[str(p) for p in pair_files]}")

    df = load_pairs_many(pair_files)
    if df.empty:
        print("loaded zero pair records — nothing to train on", file=sys.stderr)
        return 3
    print(f"Loaded {len(df)} pair records across {df['scene_id'].nunique()} scenes")
    print(f"Positive rate: {df['should_merge'].mean():.3f}")

    overrides = {}
    if args.params_json:
        with open(args.params_json) as f:
            overrides = json.load(f)

    result = train_merge_classifier(
        df,
        output_path=args.output_model,
        val_ratio=args.val_scene_ratio,
        lgb_params=overrides,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )
    write_metrics_json(result, args.output_metrics)

    print(f"\nSaved model: {result.model_path}")
    print(f"Wrote metrics: {args.output_metrics}")
    print(f"  train pairs/scenes: {result.n_pairs_train}/{result.n_scenes_train}")
    print(f"  val   pairs/scenes: {result.n_pairs_val}/{result.n_scenes_val}")
    print(f"  val accuracy: {result.val_metrics['accuracy']:.4f}")
    print(f"  val F1:       {result.val_metrics['f1']:.4f}")
    print(f"  val PR-AUC:   {result.val_metrics['pr_auc']:.4f}")
    print(f"  val ROC-AUC:  {result.val_metrics['roc_auc']:.4f}")
    print("  feature importance (gain):")
    for name, score in sorted(result.feature_importance.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: {score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
