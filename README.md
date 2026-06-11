# LidarRoofSegNet

Plane-aware neural instance segmentation of roof faces from airborne LiDAR.

## What this is

A pipeline that takes airborne LiDAR point clouds of buildings and produces
per-point **roof face** labels — one cluster ID per topological roof face
(gable side, hip, dormer, etc.) as defined in [`CONTEXT.md`](CONTEXT.md).

The headline contribution (ADR-0001) is a **learned plane-aware
cluster-graph refinement** that merges over-segmented embedding clusters into
faithful roof faces. The trainable component (ADR-0002) is a LightGBM merge
classifier trained on cluster-pair labels harvested via k-fold cross-prediction
on the training split — encoder-agnostic and reusable on top of any
over-segmenting clusterer.

```
points -> DGCNN embeddings -> HDBSCAN clusters -> plane-aware refinement -> noise recovery -> labels
                                                          ^
                                                          |
                                              LightGBM merge classifier
                                                  (or hand-tuned ablation)
```

## Headline results (roofNTNU test, n=99)

| pipeline                              |  NFE | mPrec | mIoU |  ARI |
|---------------------------------------|-----:|------:|-----:|-----:|
| raw HDBSCAN (no refinement)           | 1.21 |  0.76 | 0.92 | 0.88 |
| + hand-tuned plane scorer (ablation)  | 0.31 |  0.92 | 0.92 | 0.88 |
| **+ LightGBM @ merge_threshold=0.6**  | **0.21** | **0.94** | **0.93** | 0.88 |

Refinement cuts **number-of-faces error** by 83% vs. raw clustering; the
learned classifier wins a further 32% over the hand-tuned geometric baseline
while strictly dominating or tying it on 6 of 9 metrics. Full
{clusterer × refinement} matrix lives in `evaluation_results/ablation_matrix.csv`
after running `scripts/run_ablation_matrix.sh`.

## Quickstart

### Environment

PyTorch + PyTorch Geometric for the encoder; LightGBM for the classifier;
NumPy/scipy/sklearn for the rest. Python 3.10+.

```bash
pip install -e .[classifier]
```

The roofNTNU split lives at `data/roofNTNU/train_test_split/{points,labels}_{train,val,test}_n/`
(788 / 99 / 99 scenes). See [`docs/adr/0003-building3d-as-weak-external-validation.md`](docs/adr/0003-building3d-as-weak-external-validation.md)
for the external-validation dataset.

### End-to-end run (server, needs CUDA + PyG)

```bash
# 1. Train the encoder (legacy entry point).
python trainModel.py

# 2. Prepare k-fold split for the cross-prediction harvest.
python scripts/prepare_folds.py

# 3. Harvest cluster-pair training data (5 folds, ~8h on one GPU).
pm2 start harvest.config.js

# 4. Train the LightGBM merge classifier (CPU, ~seconds).
python scripts/train_merge_classifier.py \
    --pairs artifacts/merge_classifier/fold_*/pairs.jsonl

# 5. Evaluate (canonical pipeline + LightGBM scorer).
python scripts/evaluate.py \
    --checkpoint roof_segmentation_dgcnn_best.pth \
    --scorer-model artifacts/merge_classifier/model.txt \
    --merge-threshold 0.6 \
    --output-json evaluation_results/eval_lgb06.json
```

### Reproducibility (PM2)

Long-running multi-step jobs are packaged as PM2 configs:

| config                | what it runs                                                        |
|-----------------------|---------------------------------------------------------------------|
| `harvest.config.js`   | 5-fold cross-prediction harvest of cluster-pair training data       |
| `ablation.config.js`  | 3-row ablation: no refinement / hand-tuned / LightGBM               |
| `sweep.config.js`     | LightGBM merge-threshold sweep (0.5 / 0.6 / 0.7 / 0.8)              |
| `matrix.config.js`    | 9-row {clusterer × refinement} ablation matrix + CSV aggregator     |

### Baselines (ADR-0001 §Consequences)

```bash
# Classical normal-based region growing (no model).
python scripts/evaluate.py --baseline region_growing \
    --output-json evaluation_results/baseline_rg.json

# Closed-set softmax (DGCNN + K-way head + Hungarian-matching loss).
python scripts/train_softmax_baseline.py --num-classes 8 --epochs 50
python scripts/evaluate.py --baseline softmax \
    --softmax-checkpoint artifacts/softmax_baseline/model.pth --num-classes 8 \
    --output-json evaluation_results/baseline_softmax.json
```

### External validation on Building3D (ADR-0003)

```bash
# Project wireframes onto points to derive weak per-point labels.
python scripts/convert_building3d.py \
    --output-root data/Building3D/converted

# Run the canonical pipeline against the converted scenes.
python scripts/evaluate.py \
    --checkpoint roof_segmentation_dgcnn_best.pth \
    --scorer-model artifacts/merge_classifier/model.txt --merge-threshold 0.6 \
    --data-root data/Building3D/converted --split test \
    --output-json evaluation_results/building3d_eval.json
```

The per-scene `ambiguity_rate` in `data/Building3D/converted/audit_test.json`
is the weak-label noise floor per ADR-0003; metrics should be read in that
context.

## Project structure

```
roofseg/
  baselines/       region growing + closed-set softmax (Baseline 0)
  data/            scene loaders incl. Building3D weak-label projection
  refinement/      cluster-graph + features + scorer + greedy merge + classifier training
  cli/evaluate.py  canonical eval CLI
  inference.py     run_inference(points, embeddings|cluster_labels, ...)
  metrics.py       IoU/ARI/NMI/mIoU/mCov/mRec/mPrec + NFE + over/under-seg
  clustering.py    HDBSCAN/DBSCAN/MeanShift wrappers
trainModel.py      legacy DGCNN encoder + discriminative loss (the embedding model)
scripts/           argparse CLIs: prepare_folds, harvest_fold, train_merge_classifier,
                                  train_softmax_baseline, evaluate, convert_building3d,
                                  aggregate_ablation_matrix, run_*.sh wrappers
tests/             50+ unit tests; run with `pytest tests/`
configs/           folds.json (deterministic 5-fold split, seed=42)
docs/adr/          design decisions (read before proposing architectural changes)
```

## Design contract

These four documents anchor every design decision and override anything the
code might do or claim otherwise:

- [`CONTEXT.md`](CONTEXT.md) — terminology; the canonical noun is **roof face**.
- [`Project_vision_v1.md`](Project_vision_v1.md) — paper thesis, target metrics, roadmap.
- [`docs/adr/0001-refinement-as-core-contribution.md`](docs/adr/0001-refinement-as-core-contribution.md) — plane-aware refinement is the contribution.
- [`docs/adr/0002-learned-merge-classifier.md`](docs/adr/0002-learned-merge-classifier.md) — the merge rule is learned (LightGBM), not hand-tuned.
- [`docs/adr/0003-building3d-as-weak-external-validation.md`](docs/adr/0003-building3d-as-weak-external-validation.md) — Building3D is weak-label external validation only.

## Status

The headline result (Section above) is reproducible end-to-end from the data
+ this repo. Remaining gaps before submission: baseline numbers from the
ablation matrix on the server, Building3D external-validation table, paper
draft.
