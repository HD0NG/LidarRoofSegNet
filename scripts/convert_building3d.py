#!/usr/bin/env python3
"""Convert Building3D Entry-level scenes into roofNTNU on-disk format (ADR-0003).

For each (xyz, obj) pair: load + project weak per-point labels via the
planar-face finder, center+unit-sphere normalize, then write XYZ + integer
labels in roofNTNU's ``points_{split}_n/{scene_id}.txt`` +
``labels_{split}_n/{scene_id}.txt`` layout so the existing eval CLI runs
against the converted data unchanged.

Also writes a per-scene audit JSON (``n_faces``, ``ambiguity_rate``,
``n_assigned``) so the external-validation eval can quantify weak-label
noise per ADR-0003.

Scenes whose weak labels exceed ``--max-ambiguity-rate`` are rejected,
matching the "ambiguity-filtered external validation" the ADR requires.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Building3D Entry-level to roofNTNU format.")
    p.add_argument("--input-root", default="data/Building3D/Entry-level/train",
                   help="dir with xyz/ and wireframe/ subdirectories")
    p.add_argument("--output-root", default="data/Building3D/converted",
                   help="dir to write points_{split}_n/ and labels_{split}_n/")
    p.add_argument("--split-name", default="test",
                   help="suffix on output dirs (matches eval CLI's --split)")
    p.add_argument("--max-scenes", type=int, default=None,
                   help="random subsample of input scenes (default: all)")
    p.add_argument("--ambiguity-margin", type=float, default=0.3,
                   help="weak-label ambiguity threshold (raw building units)")
    p.add_argument("--max-ambiguity-rate", type=float, default=0.10,
                   help="reject scenes whose weak labels are noisier than this")
    p.add_argument("--min-faces", type=int, default=1,
                   help="reject scenes with fewer faces than this")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from roofseg.data.building3d import load_scene

    in_root = Path(args.input_root)
    xyz_dir = in_root / "xyz"
    obj_dir = in_root / "wireframe"
    if not xyz_dir.exists() or not obj_dir.exists():
        print(f"missing xyz/ or wireframe/ under {in_root}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    ids = sorted(p.stem for p in xyz_dir.glob("*.xyz"))
    ids = [i for i in ids if (obj_dir / f"{i}.obj").exists()]
    if args.max_scenes and args.max_scenes < len(ids):
        ids = rng.sample(ids, args.max_scenes)
        ids.sort()

    out_root = Path(args.output_root)
    points_dir = out_root / f"points_{args.split_name}_n"
    labels_dir = out_root / f"labels_{args.split_name}_n"
    points_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    audit: list[dict] = []
    n_ok = n_fail = n_rejected = 0
    for sid in ids:
        try:
            scene = load_scene(
                sid, xyz_dir / f"{sid}.xyz", obj_dir / f"{sid}.obj",
                ambiguity_margin=args.ambiguity_margin,
            )
        except Exception as exc:  # broad: malformed obj / xyz happens in the wild
            n_fail += 1
            print(f"  FAIL {sid}: {exc}")
            continue

        if scene.n_faces < args.min_faces or scene.ambiguity_rate > args.max_ambiguity_rate:
            n_rejected += 1
            continue

        pts = scene.points - scene.points.mean(axis=0)
        scale = float(np.linalg.norm(pts, axis=1).max())
        if scale > 0:
            pts = pts / scale

        np.savetxt(points_dir / f"{sid}.txt", pts, fmt="%.6f")
        np.savetxt(labels_dir / f"{sid}.txt", scene.labels.astype(np.int64), fmt="%d")
        audit.append({
            "scene_id": sid,
            "n_points": int(pts.shape[0]),
            "n_faces": int(scene.n_faces),
            "n_assigned": int((scene.labels >= 0).sum()),
            "n_ambiguous": int(scene.n_ambiguous),
            "ambiguity_rate": float(scene.ambiguity_rate),
            "scale": scale,
        })
        n_ok += 1

    summary = {
        "n_scenes_input": len(ids),
        "n_scenes_ok": n_ok,
        "n_scenes_rejected": n_rejected,
        "n_scenes_failed": n_fail,
        "ambiguity_margin": args.ambiguity_margin,
        "max_ambiguity_rate": args.max_ambiguity_rate,
        "min_faces": args.min_faces,
        "mean_ambiguity_rate": float(np.mean([a["ambiguity_rate"] for a in audit])) if audit else 0.0,
        "mean_n_faces": float(np.mean([a["n_faces"] for a in audit])) if audit else 0.0,
        "scenes": audit,
    }
    audit_path = out_root / f"audit_{args.split_name}.json"
    with open(audit_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nConverted {n_ok} scenes (failed={n_fail}, rejected={n_rejected})")
    print(f"  output dir: {out_root}")
    print(f"  audit:      {audit_path}")
    print(f"  mean ambiguity rate: {summary['mean_ambiguity_rate'] * 100:.2f}%")
    print(f"  mean n_faces:        {summary['mean_n_faces']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
