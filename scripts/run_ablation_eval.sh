#!/usr/bin/env bash
set -e
export MKL_THREADING_LAYER=GNU

PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
CHECKPOINT="${CHECKPOINT:-roof_segmentation_dgcnn_best.pth}"
OUT_DIR="${OUT_DIR:-evaluation_results}"

# Row 1: raw clustering only (no refinement).
"$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" --no-refinement \
    --output-json "$OUT_DIR/ablation_raw.json"

# Row 2: + hand-tuned plane-aware scorer (baseline refinement).
"$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --output-json "$OUT_DIR/ablation_handtuned.json"

# Row 3: + trained LightGBM merge classifier (headline configuration).
"$PYTHON" scripts/evaluate.py --checkpoint "$CHECKPOINT" \
    --scorer-model artifacts/merge_classifier/model.txt \
    --output-json "$OUT_DIR/ablation_lightgbm.json"

exit 0
