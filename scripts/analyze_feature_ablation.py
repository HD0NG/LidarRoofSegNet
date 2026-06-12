#!/usr/bin/env python3
"""Feature-subset ablation for the LightGBM merge classifier (ADR-0002).

The full model's gain importance is dominated by ``f_joint_residual`` (~5x the
next feature). This asks: how much of the full 11-feature performance does a
parsimonious top-k model recover? A small subset matching the full model is a
strong interpretability + anti-overfitting result for the paper.

Reuses the exact scene-level split and metric helpers from
``classifier_training`` so numbers are directly comparable to the released
model. Pure CPU / lightgbm — runs locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roofseg.refinement.classifier_training import (
    DEFAULT_LGB_PARAMS,
    _binary_metrics,
    split_scenes,
)
from roofseg.refinement.training_data import feature_columns, load_pairs_many

# Features ranked by the released full model's gain importance (metrics.json).
RANKED_FEATURES = [
    "f_joint_residual",
    "f_aspect_diff",
    "f_slope_diff",
    "f_normal_angle",
    "f_residual_a",
    "f_size_ratio",
    "f_size_a",
    "f_size_b",
    "f_boundary_distance",
    "f_residual_b",
    "f_offset_diff",
]

TOPK = [1, 2, 3, 5, 11]


def _best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1 at the score-maximising threshold (threshold-free comparison).

    The fixed-0.5 F1 is misleading across subset models: a well-ranked but
    under-confident model scores nothing above 0.5 and collapses to F1=0. The
    best-over-thresholds F1 reflects ranking quality the pipeline can exploit
    by tuning its merge threshold.
    """
    from sklearn.metrics import f1_score

    thresholds = np.unique(y_score)
    best = 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        best = max(best, float(f1))
    return best


def _train_subset(df, feat_cols, *, seed=42):
    import lightgbm as lgb

    train_scenes, val_scenes = split_scenes(
        df["scene_id"].unique().tolist(), val_ratio=0.2, seed=seed
    )
    train_df = df[df["scene_id"].isin(train_scenes)]
    val_df = df[df["scene_id"].isin(val_scenes)]

    X_train = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_train = train_df["should_merge"].astype(int).to_numpy()
    X_val = val_df[feat_cols].to_numpy(dtype=np.float64)
    y_val = val_df["should_merge"].astype(int).to_numpy()

    params = {**DEFAULT_LGB_PARAMS, "seed": seed}
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    val_scores = booster.predict(X_val)
    m = _binary_metrics(y_val, val_scores)
    m["best_f1"] = _best_f1(y_val, val_scores)
    return m


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LightGBM feature-subset ablation.")
    p.add_argument(
        "--pairs", nargs="+",
        default=[f"artifacts/merge_classifier/fold_{k}/pairs.jsonl" for k in range(5)],
    )
    p.add_argument("--output-json", default="artifacts/merge_classifier/feature_ablation.json")
    args = p.parse_args(argv)

    all_feats = feature_columns()
    assert set(RANKED_FEATURES) == set(all_feats), "ranked list must match FEATURE_ORDER"

    df = load_pairs_many([Path(p) for p in args.pairs])
    print(f"Loaded {len(df)} pairs across {df['scene_id'].nunique()} scenes\n")

    results = []
    for k in TOPK:
        feats = RANKED_FEATURES[:k]
        m = _train_subset(df, feats)
        results.append({"k": k, "features": feats, "val_metrics": m})

    full = next(r for r in results if r["k"] == 11)["val_metrics"]
    out = {"results": results, "full_reference": full}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'top-k':>6} {'PR-AUC':>8} {'ROC-AUC':>8} {'best-F1':>8}  {'% of full PR-AUC':>16}")
    print("-" * 56)
    for r in results:
        m = r["val_metrics"]
        pct = 100.0 * m["pr_auc"] / full["pr_auc"]
        print(f"{r['k']:>6} {m['pr_auc']:>8.3f} {m['roc_auc']:>8.3f} {m['best_f1']:>8.3f}  {pct:>15.1f}%")
    print("\n(PR-AUC / ROC-AUC are threshold-free; best-F1 is over all thresholds.)")
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
