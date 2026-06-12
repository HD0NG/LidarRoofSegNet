#!/usr/bin/env python3
"""Render qualitative segmentation figures from dumped predictions (local).

Reads the ``.npz`` files written by ``scripts/dump_predictions.py`` and renders
a grid: one row per scene, columns = [ground truth, raw clustering, hand-tuned
refinement, LightGBM refinement]. Roof scenes are near-2.5D, so panels are
top-down (XY) scatter plots coloured by instance label. Pure matplotlib — no
PyG, runs locally once the dumps are pulled from the server.

Colour convention: each panel colours its own instances with a fixed
qualitative colormap, labels remapped to 0..K-1 by descending cluster size for
visual stability. Over-segmentation reads as a GT region appearing in extra
colours; noise (-1) is light grey. Panel titles show the instance count so the
number-of-faces error is legible at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLUMNS = [("gt", "Ground truth"), ("raw", "Raw clustering"),
           ("handtuned", "Hand-tuned"), ("lgb", "LightGBM")]
_CMAP = plt.get_cmap("tab20")


def _size_ordered(labels: np.ndarray) -> np.ndarray:
    """Remap labels to 0..K-1 by descending size; keep -1 as -1."""
    out = np.full_like(labels, -1)
    ids, counts = np.unique(labels[labels != -1], return_counts=True)
    for new_id, old_id in enumerate(ids[np.argsort(-counts)]):
        out[labels == old_id] = new_id
    return out


def _scatter(ax, xy, labels, title):
    labels = _size_ordered(labels)
    noise = labels == -1
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=3, c="#d9d9d9", linewidths=0)
    for k in range(labels.max() + 1 if labels.max() >= 0 else 0):
        m = labels == k
        ax.scatter(xy[m, 0], xy[m, 1], s=4, color=_CMAP(k % 20), linewidths=0)
    n = int(np.unique(labels[labels != -1]).size)
    ax.set_title(f"{title}\n({n} face{'s' if n != 1 else ''})", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render qualitative segmentation grid.")
    p.add_argument("--input-dir", default="evaluation_results/qualitative")
    p.add_argument("--scenes", type=int, nargs="+", default=None,
                   help="scene indices to include; default: all .npz in input-dir")
    p.add_argument("--out-prefix", default="paper/figures/qualitative")
    args = p.parse_args(argv)

    in_dir = Path(args.input_dir)
    if args.scenes is not None:
        paths = [in_dir / f"scene_{i:03d}.npz" for i in args.scenes]
    else:
        paths = sorted(in_dir.glob("scene_*.npz"))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"no scene dumps found in {in_dir}")
        return 2

    n_rows = len(paths)
    fig, axes = plt.subplots(n_rows, len(COLUMNS),
                             figsize=(3 * len(COLUMNS), 3 * n_rows),
                             squeeze=False)
    for r, path in enumerate(paths):
        d = np.load(path)
        # Restrict to real points (gt != -1); padding points are clustered by
        # the model but excluded from metrics, so plotting/counting them would
        # inflate face counts relative to the reported tables.
        valid = d["gt"] != -1
        xy = d["points"][valid][:, :2]
        for c, (key, title) in enumerate(COLUMNS):
            _scatter(axes[r][c], xy, d[key][valid], title)
        axes[r][0].set_ylabel(path.stem, fontsize=9)

    fig.tight_layout()
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    # Strip the nondeterministic CreationDate so the tracked PDF is byte-stable.
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    print(f"Wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')} ({n_rows} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
