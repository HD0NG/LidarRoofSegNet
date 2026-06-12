#!/usr/bin/env python3
"""Paired significance tests between pipeline configurations (per-scene).

Reads the per-scene metric JSONs written by ``roofseg.cli.evaluate`` and runs
paired Wilcoxon signed-rank tests on matched scenes, plus win/loss/tie counts
and a bootstrap CI on the mean paired difference. Pure NumPy/scipy — runs
locally without PyG.

The three comparisons that matter for the paper:
  * raw vs hand-tuned     — does plane-aware refinement help at all?
  * hand-tuned vs LightGBM — does the learned classifier beat the hand-tuned
                             geometric baseline on the same features? (ADR-0002)
  * raw vs LightGBM       — end-to-end effect of learned refinement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

# Metrics where a LOWER value is better (everything else: higher is better).
LOWER_IS_BETTER = {
    "number_of_faces_error",
    "over_segmentation_rate",
    "under_segmentation_rate",
}
HEADLINE_METRICS = [
    "number_of_faces_error",
    "ARI",
    "NMI",
    "mIoU",
    "mPrec",
    "under_segmentation_rate",
]


def _load(tag: str, results_dir: Path) -> list[dict]:
    return json.load(open(results_dir / f"{tag}.json"))


def _paired_arrays(a_rows, b_rows, metric):
    a = np.array([r[metric] for r in a_rows], dtype=np.float64)
    b = np.array([r[metric] for r in b_rows], dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask]


def _bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    if diff.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a_rows, b_rows, metric):
    """Compare config A to config B on one metric. Improvement = B is better than A."""
    a, b = _paired_arrays(a_rows, b_rows, metric)
    delta = b - a  # raw change from A to B
    if metric in LOWER_IS_BETTER:
        improvement = a - b  # positive => B improved (lower)
    else:
        improvement = b - a  # positive => B improved (higher)

    n_better = int((improvement > 0).sum())
    n_worse = int((improvement < 0).sum())
    n_tie = int((improvement == 0).sum())

    try:
        stat, p = wilcoxon(a, b)
        p = float(p)
    except ValueError:
        stat, p = float("nan"), float("nan")  # all differences zero

    lo, hi = _bootstrap_ci(delta)
    return {
        "metric": metric,
        "n": int(a.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_delta_b_minus_a": float(delta.mean()),
        "mean_delta_ci95": [lo, hi],
        "median_improvement": float(np.median(improvement)),
        "n_better": n_better,
        "n_worse": n_worse,
        "n_tie": n_tie,
        "wilcoxon_p": p,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paired significance tests across pipeline configs.")
    p.add_argument("--results-dir", default="evaluation_results")
    p.add_argument("--output-json", default="evaluation_results/significance_tests.json")
    args = p.parse_args(argv)

    rdir = Path(args.results_dir)
    raw = _load("ablation_raw", rdir)
    handtuned = _load("ablation_handtuned", rdir)
    lgb = _load("sweep_thresh_060", rdir)  # LightGBM @ 0.6

    comparisons = {
        "raw_vs_handtuned": (raw, handtuned),
        "handtuned_vs_lgb06": (handtuned, lgb),
        "raw_vs_lgb06": (raw, lgb),
    }

    out: dict[str, list] = {}
    for name, (a_rows, b_rows) in comparisons.items():
        out[name] = [compare(a_rows, b_rows, m) for m in HEADLINE_METRICS]

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    for name, rows in out.items():
        a_label, b_label = name.split("_vs_")
        print(f"\n=== {a_label}  ->  {b_label}  (improvement = {b_label} better) ===")
        print(f"{'metric':<26} {'mean_A':>8} {'mean_B':>8} {'W/L/T':>12} {'p':>10}  CI95(B-A)")
        print("-" * 86)
        for r in rows:
            wlt = f"{r['n_better']}/{r['n_worse']}/{r['n_tie']}"
            lo, hi = r["mean_delta_ci95"]
            sig = "***" if r["wilcoxon_p"] < 0.001 else "**" if r["wilcoxon_p"] < 0.01 else "*" if r["wilcoxon_p"] < 0.05 else "ns"
            pstr = f"{r['wilcoxon_p']:.2e}" if r["wilcoxon_p"] == r["wilcoxon_p"] else "n/a"
            print(
                f"{r['metric']:<26} {r['mean_a']:>8.3f} {r['mean_b']:>8.3f} {wlt:>12} "
                f"{pstr:>10} {sig:>3}  [{lo:+.3f}, {hi:+.3f}]"
            )

    print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
