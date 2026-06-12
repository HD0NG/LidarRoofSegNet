#!/usr/bin/env bash
# Building3D external validation (ADR-0003): convert wireframes -> weak
# per-point labels in roofNTNU format, then evaluate the roofNTNU-trained
# pipeline on them. Self-contained: runs the conversion on the server from the
# raw Building3D data (data/ is gitignored, so converted data does not travel
# over git). Deterministic (seed=42) so it matches the local preview.
#
# The three rows mirror the roofNTNU ablation so the transfer comparison is
# apples-to-apples: raw clustering / hand-tuned scorer / LightGBM@0.6. The
# hand-tuned tolerances are calibrated to roofNTNU's unit-sphere scale; the
# learned classifier operates on geometric features. Whether the learned model
# transfers better is the question the roofNTNU test could not answer
# significantly (see significance_tests.json).
set -e
export MKL_THREADING_LAYER=GNU

PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
DGCNN_CKPT="${DGCNN_CKPT:-roof_segmentation_dgcnn_best.pth}"
LGB="${LGB:-artifacts/merge_classifier/model.txt}"
MERGE_THRESH="${MERGE_THRESH:-0.6}"
OUT_DIR="${OUT_DIR:-evaluation_results}"

# Conversion knobs.
B3D_INPUT="${B3D_INPUT:-data/Building3D/Entry-level/train}"
B3D_CONVERTED="${B3D_CONVERTED:-data/Building3D/converted}"
MAX_SCENES="${MAX_SCENES:-800}"
MIN_FACES="${MIN_FACES:-2}"
MAX_AMBIGUITY="${MAX_AMBIGUITY:-0.10}"

echo "=== [1/4] Converting Building3D wireframes -> weak labels ==="
"$PYTHON" scripts/convert_building3d.py \
    --input-root "$B3D_INPUT" \
    --output-root "$B3D_CONVERTED" \
    --split-name test \
    --max-scenes "$MAX_SCENES" \
    --min-faces "$MIN_FACES" \
    --max-ambiguity-rate "$MAX_AMBIGUITY"

eval_row() {
    local tag="$1"; shift
    "$PYTHON" scripts/evaluate.py \
        --checkpoint "$DGCNN_CKPT" \
        --data-root "$B3D_CONVERTED" --split test \
        --output-json "$OUT_DIR/building3d_${tag}.json" "$@"
}

echo "=== [2/4] raw clustering (no refinement) ==="
eval_row raw --no-refinement

echo "=== [3/4] hand-tuned plane-aware refinement ==="
eval_row handtuned

echo "=== [4/4] LightGBM @ ${MERGE_THRESH} (transfer test) ==="
eval_row lgb06 --scorer-model "$LGB" --merge-threshold "$MERGE_THRESH"

echo ""
echo "Done. External-validation summaries:"
echo "  $OUT_DIR/building3d_{raw,handtuned,lgb06}_summary.json"
echo "Weak-label audit (read alongside the metrics per ADR-0003):"
echo "  $B3D_CONVERTED/audit_test.json"
exit 0
