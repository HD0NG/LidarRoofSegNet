#!/usr/bin/env python3
"""Aggregate the 9-row ablation matrix into one CSV + a printed table.

Reads ``matrix_{clusterer}_{refinement}_summary.json`` files produced by
``scripts/run_ablation_matrix.sh`` and joins them into a single comparison
table for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROWS = [
    ("dgcnn", "off"),
    ("dgcnn", "handtuned"),
    ("dgcnn", "lgb06"),
    ("rg", "off"),
    ("rg", "handtuned"),
    ("rg", "lgb06"),
    ("softmax", "off"),
    ("softmax", "handtuned"),
    ("softmax", "lgb06"),
]

METRICS = [
    "mean_ARI",
    "mean_NMI",
    "mean_mIoU",
    "mean_mCov",
    "mean_mWCov",
    "mean_mRec",
    "mean_mPrec",
    "mean_mWPrec",
    "mean_number_of_faces_error",
    "mean_over_segmentation_rate",
    "mean_under_segmentation_rate",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate ablation matrix summaries.")
    p.add_argument("--input-dir", default="evaluation_results")
    p.add_argument("--output-csv", default="evaluation_results/ablation_matrix.csv")
    args = p.parse_args(argv)

    in_dir = Path(args.input_dir)
    rows_out: list[dict] = []
    for clusterer, refinement in ROWS:
        tag = f"{clusterer}_{refinement}"
        path = in_dir / f"matrix_{tag}_summary.json"
        if not path.exists():
            print(f"missing: {path}")
            continue
        with open(path) as f:
            summary = json.load(f)
        row = {"clusterer": clusterer, "refinement": refinement}
        for m in METRICS:
            row[m] = summary.get(m, "")
        rows_out.append(row)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["clusterer", "refinement"] + METRICS
        )
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"\nWrote {out_path}")
    print(
        f"\n{'clusterer':<10} {'refinement':<12} "
        f"{'NFE':>7} {'ARI':>7} {'mIoU':>7} {'mPrec':>7} {'over':>7} {'under':>7}"
    )
    print("-" * 72)
    for row in rows_out:
        def fmt(key: str) -> str:
            v = row.get(key, "")
            return f"{float(v):>7.3f}" if v != "" else "    n/a"

        print(
            f"{row['clusterer']:<10} {row['refinement']:<12} "
            f"{fmt('mean_number_of_faces_error')} "
            f"{fmt('mean_ARI')} "
            f"{fmt('mean_mIoU')} "
            f"{fmt('mean_mPrec')} "
            f"{fmt('mean_over_segmentation_rate')} "
            f"{fmt('mean_under_segmentation_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
