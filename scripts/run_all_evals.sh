#!/usr/bin/env bash
# Bit-reproducible re-run of every paper eval through the now-deterministic
# pipeline (per-scene FPS seeding, trainModel.py). Sequences the existing
# wrappers so all tables + figures come from one deterministic pass and agree
# to the integer. Re-run any time to regenerate the full results set.
#
# Each sub-wrapper runs as its own process (its internal `exit 0` exits the
# child, not this batch). Shared env is exported so the children inherit it.
set -e
export MKL_THREADING_LAYER=GNU
export PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
export OUT_DIR="${OUT_DIR:-evaluation_results}"
export DGCNN_CKPT="${DGCNN_CKPT:-roof_segmentation_dgcnn_best.pth}"
export CHECKPOINT="${CHECKPOINT:-$DGCNN_CKPT}"   # run_threshold_sweep.sh uses CHECKPOINT

echo "=== [1/5] main ablation + LightGBM threshold sweep (test split) ==="
bash scripts/run_threshold_sweep.sh

echo "=== [2/5] LightGBM threshold sweep (val split) ==="
bash scripts/run_val_threshold_sweep.sh

echo "=== [3/5] 9-row {clusterer x refinement} ablation matrix + CSV ==="
bash scripts/run_ablation_matrix.sh

echo "=== [4/5] Building3D external validation (convert + 3 rows) ==="
bash scripts/run_building3d_eval.sh

echo "=== [5/5] qualitative prediction dumps ==="
"$PYTHON" scripts/dump_predictions.py --output-dir "$OUT_DIR/qualitative"

echo ""
echo "All evals reproduced deterministically. Tables + figure now share one pipeline."
exit 0
