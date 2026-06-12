#!/usr/bin/env bash
# Validation-split threshold selection (experimental-design fix).
#
# The headline LightGBM operating point (merge_threshold=0.6) was originally
# picked from a sweep on the TEST split — a test-set-tuning criticism. This
# re-runs the LightGBM sweep on the VAL split so the operating point can be
# SELECTED on val and only then reported on test (those test numbers already
# exist in evaluation_results/sweep_thresh_*.json).
#
# After this runs, pick argmin-NFE (or the best composite) on val from
# sweep_val_thresh_*_summary.json, then report the matching test row.
set -e
export MKL_THREADING_LAYER=GNU

PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
CHECKPOINT="${CHECKPOINT:-roof_segmentation_dgcnn_best.pth}"
LGB="${LGB:-artifacts/merge_classifier/model.txt}"
OUT_DIR="${OUT_DIR:-evaluation_results}"

THRESHOLDS=(0.5 0.6 0.7 0.8)
TAGS=(050 060 070 080)

for i in 0 1 2 3; do
  t="${THRESHOLDS[$i]}"
  tag="${TAGS[$i]}"
  "$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" \
      --split val \
      --scorer-model "$LGB" \
      --merge-threshold "$t" \
      --output-json "$OUT_DIR/sweep_val_thresh_${tag}.json"
done

exit 0
