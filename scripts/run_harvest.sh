#!/usr/bin/env bash
set -e
export MKL_THREADING_LAYER=GNU
for k in 0 1 2 3 4; do
  /home/ubuntu/miniconda3/envs/LiDARML/bin/python \
    scripts/harvest_fold.py --fold $k \
    --epochs 50 --output-dir artifacts/merge_classifier
done
exit 0
