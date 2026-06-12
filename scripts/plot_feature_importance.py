#!/usr/bin/env python3
"""Feature-importance + parsimony figure for the merge classifier (ADR-0002).

Two panels:
  (a) LightGBM gain importance per feature (horizontal bars). f_joint_residual
      dominates by ~8.5x — the figure is meant to show that dominance.
  (b) Feature-subset ablation: val PR-AUC vs number of top-k features, with the
      full-model reference line. Shows 3 features recover ~94% of full PR-AUC.

Reads ``artifacts/merge_classifier/metrics.json`` and
``artifacts/merge_classifier/feature_ablation.json``. Pure matplotlib — no PyG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pretty(name: str) -> str:
    return name.removeprefix("f_").replace("_", " ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plot merge-classifier feature importance + parsimony.")
    p.add_argument("--metrics", default="artifacts/merge_classifier/metrics.json")
    p.add_argument("--ablation", default="artifacts/merge_classifier/feature_ablation.json")
    p.add_argument("--out-prefix", default="paper/figures/feature_importance")
    args = p.parse_args(argv)

    fi = json.load(open(args.metrics))["feature_importance"]
    items = sorted(fi.items(), key=lambda kv: kv[1])  # ascending for barh
    names = [_pretty(k) for k, _ in items]
    values = [v for _, v in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): gain importance.
    colors = ["#c44e52" if n == "joint residual" else "#4c72b0" for n in names]
    ax1.barh(names, values, color=colors)
    ax1.set_xlabel("LightGBM gain importance")
    ax1.set_title("(a) Feature importance")
    for y, v in enumerate(values):
        ax1.text(v + max(values) * 0.01, y, f"{v:,.0f}", va="center", fontsize=8)
    ax1.set_xlim(0, max(values) * 1.15)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel (b): parsimony curve.
    abl = json.load(open(args.ablation))
    ks = [r["k"] for r in abl["results"]]
    pr = [r["val_metrics"]["pr_auc"] for r in abl["results"]]
    full = abl["full_reference"]["pr_auc"]
    ax2.plot(ks, pr, marker="o", color="#4c72b0", label="top-k subset")
    ax2.axhline(full, ls="--", color="#c44e52", lw=1, label=f"full model ({full:.3f})")
    ax2.set_xlabel("number of top-k features")
    ax2.set_ylabel("validation PR-AUC")
    ax2.set_title("(b) Parsimony: PR-AUC vs feature count")
    ax2.set_xticks(ks)
    for k, v in zip(ks, pr):
        ax2.annotate(f"{100*v/full:.0f}%", (k, v), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color="#555")
    ax2.set_ylim(min(pr) * 0.96, full * 1.03)
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")  # vector for LaTeX
    print(f"Wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
