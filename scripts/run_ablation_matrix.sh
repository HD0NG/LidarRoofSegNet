#!/usr/bin/env bash
# ADR-0001 §Consequences: every baseline gets evaluated with AND without
# plane-aware refinement. The matrix is {clusterer} x {refinement}.
set -e
export MKL_THREADING_LAYER=GNU

PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/LiDARML/bin/python}"
DGCNN_CKPT="${DGCNN_CKPT:-roof_segmentation_dgcnn_best.pth}"
SOFTMAX_CKPT="${SOFTMAX_CKPT:-artifacts/softmax_baseline/model.pth}"
NUM_CLASSES="${NUM_CLASSES:-8}"
OUT_DIR="${OUT_DIR:-evaluation_results}"
LGB="${LGB:-artifacts/merge_classifier/model.txt}"
MERGE_THRESH="${MERGE_THRESH:-0.6}"
# Which clusterer groups to run: "all" or any subset of {dgcnn,rg,softmax}
# (space- or comma-separated). Re-aggregation always runs and reuses whatever
# summaries already exist, so e.g. ONLY=rg refreshes just the 3 RG rows.
ONLY="${ONLY:-all}"

run_row() {
    local tag="$1"; shift
    "$PYTHON" scripts/evaluate.py \
        --output-json "$OUT_DIR/matrix_${tag}.json" "$@"
}

should_run() {
    [[ "$ONLY" == "all" ]] && return 0
    [[ " ${ONLY//,/ } " == *" $1 "* ]] && return 0
    return 1
}

# DGCNN embeddings + HDBSCAN (the canonical pipeline).
if should_run dgcnn; then
    run_row dgcnn_off       --checkpoint "$DGCNN_CKPT" --no-refinement
    run_row dgcnn_handtuned --checkpoint "$DGCNN_CKPT"
    run_row dgcnn_lgb06     --checkpoint "$DGCNN_CKPT" --scorer-model "$LGB" --merge-threshold "$MERGE_THRESH"
fi

# Classical region growing (pre-deep-learning baseline).
if should_run rg; then
    run_row rg_off       --baseline region_growing --no-refinement
    run_row rg_handtuned --baseline region_growing
    run_row rg_lgb06     --baseline region_growing --scorer-model "$LGB" --merge-threshold "$MERGE_THRESH"
fi

# Closed-set softmax (Baseline 0).
if should_run softmax; then
    run_row softmax_off       --baseline softmax --softmax-checkpoint "$SOFTMAX_CKPT" --num-classes "$NUM_CLASSES" --no-refinement
    run_row softmax_handtuned --baseline softmax --softmax-checkpoint "$SOFTMAX_CKPT" --num-classes "$NUM_CLASSES"
    run_row softmax_lgb06     --baseline softmax --softmax-checkpoint "$SOFTMAX_CKPT" --num-classes "$NUM_CLASSES" --scorer-model "$LGB" --merge-threshold "$MERGE_THRESH"
fi

"$PYTHON" scripts/aggregate_ablation_matrix.py \
    --input-dir "$OUT_DIR" \
    --output-csv "$OUT_DIR/ablation_matrix.csv"

exit 0
