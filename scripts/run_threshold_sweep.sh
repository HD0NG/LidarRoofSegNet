#!/usr/bin/env bash
set -e
export MKL_THREADING_LAYER=GNU

PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
CHECKPOINT="${CHECKPOINT:-roof_segmentation_dgcnn_best.pth}"
OUT_DIR="${OUT_DIR:-evaluation_results}"

# Row 1 + Row 2 re-runs so all rows share the fixed coverage-based
# over/under-segmentation metric. (Row 3 is reproduced by sweep_thresh_050.)
"$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" --no-refinement \
    --output-json "$OUT_DIR/ablation_raw.json"

"$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --output-json "$OUT_DIR/ablation_handtuned.json"

# LightGBM merge-threshold sweep.
THRESHOLDS=(0.5 0.6 0.7 0.8)
TAGS=(050 060 070 080)

for i in 0 1 2 3; do
  t="${THRESHOLDS[$i]}"
  tag="${TAGS[$i]}"
  "$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" \
      --scorer-model artifacts/merge_classifier/model.txt \
      --merge-threshold "$t" \
      --output-json "$OUT_DIR/sweep_thresh_${tag}.json"
done

# Row 3 is LightGBM @ 0.5 — mirror so all three ablation files share the fixed metric.
cp "$OUT_DIR/sweep_thresh_050.json"         "$OUT_DIR/ablation_lightgbm.json"
cp "$OUT_DIR/sweep_thresh_050_summary.json" "$OUT_DIR/ablation_lightgbm_summary.json"

exit 0
