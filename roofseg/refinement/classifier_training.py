"""Train the LightGBM merge classifier on harvested pair records (ADR-0002).

Inputs are JSON-lines files produced by :mod:`roofseg.refinement.training_data`.
Outputs are a saved LightGBM booster (``.txt``) plus a metrics JSON.

Splitting:
    Train/validation split is **by scene**, not by pair, to avoid leakage —
    two adjacent pairs from the same scene share point geometry. The
    classifier must generalise across buildings.

Class imbalance:
    Roof scenes typically have more "should-not-merge" pairs than "should-merge"
    pairs (most adjacent clusters straddle a ridge). We pass
    ``is_unbalance=True`` to LightGBM by default and report PR-AUC alongside
    accuracy so reviewers can judge the classifier on the under-represented
    class.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roofseg.refinement.training_data import feature_columns


@dataclass
class ClassifierTrainResult:
    model_path: str
    n_pairs_total: int
    n_pairs_train: int
    n_pairs_val: int
    n_scenes_train: int
    n_scenes_val: int
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    lgb_params: dict[str, Any] = field(default_factory=dict)


def split_scenes(scene_ids: list[str], *, val_ratio: float = 0.2, seed: int = 42) -> tuple[set[str], set[str]]:
    """Deterministically split scene ids into train/val sets."""
    sorted_ids = sorted(set(scene_ids))
    rng = random.Random(seed)
    rng.shuffle(sorted_ids)
    n_val = max(1, int(round(len(sorted_ids) * val_ratio)))
    val_set = set(sorted_ids[:n_val])
    train_set = set(sorted_ids[n_val:])
    return train_set, val_set


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, PR-AUC, ROC-AUC, plus positive-rate sanity counts."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_score >= threshold).astype(np.int64)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "n": int(y_true.size),
    }
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc", "average_precision"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "is_unbalance": True,
    "verbose": -1,
}


def train_merge_classifier(
    df,
    *,
    output_path: str | Path,
    val_ratio: float = 0.2,
    lgb_params: dict[str, Any] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    seed: int = 42,
) -> ClassifierTrainResult:
    """Train LightGBM on the harvested pairs DataFrame.

    ``df`` must contain a ``scene_id`` column, a boolean ``should_merge`` column,
    and the feature columns ``f_<name>`` produced by
    :func:`roofseg.refinement.training_data.feature_columns`.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:  # pragma: no cover - covered by install instructions
        raise RuntimeError(
            "lightgbm is required to train the merge classifier. "
            "Install with `pip install lightgbm` or `pip install -e .[classifier]`."
        ) from e

    feat_cols = feature_columns()
    missing = set(feat_cols) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing feature columns: {sorted(missing)}")
    if "scene_id" not in df.columns or "should_merge" not in df.columns:
        raise ValueError("DataFrame must contain 'scene_id' and 'should_merge' columns")

    train_scenes, val_scenes = split_scenes(
        df["scene_id"].unique().tolist(), val_ratio=val_ratio, seed=seed
    )
    train_df = df[df["scene_id"].isin(train_scenes)]
    val_df = df[df["scene_id"].isin(val_scenes)]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"empty split: train_pairs={len(train_df)} val_pairs={len(val_df)}; "
            "need more scenes or a smaller val_ratio"
        )

    X_train = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_train = train_df["should_merge"].astype(int).to_numpy()
    X_val = val_df[feat_cols].to_numpy(dtype=np.float64)
    y_val = val_df["should_merge"].astype(int).to_numpy()

    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {}), "seed": seed}

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feat_cols, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_path))

    train_scores = booster.predict(X_train)
    val_scores = booster.predict(X_val)
    train_metrics = _binary_metrics(y_train, train_scores)
    val_metrics = _binary_metrics(y_val, val_scores)

    importance_gain = booster.feature_importance(importance_type="gain")
    importance = {name: float(score) for name, score in zip(feat_cols, importance_gain)}

    return ClassifierTrainResult(
        model_path=str(output_path),
        n_pairs_total=len(df),
        n_pairs_train=len(train_df),
        n_pairs_val=len(val_df),
        n_scenes_train=len(train_scenes),
        n_scenes_val=len(val_scenes),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        feature_importance=importance,
        lgb_params=params,
    )


def write_metrics_json(result: ClassifierTrainResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": result.model_path,
        "n_pairs_total": result.n_pairs_total,
        "n_pairs_train": result.n_pairs_train,
        "n_pairs_val": result.n_pairs_val,
        "n_scenes_train": result.n_scenes_train,
        "n_scenes_val": result.n_scenes_val,
        "train_metrics": result.train_metrics,
        "val_metrics": result.val_metrics,
        "feature_importance": result.feature_importance,
        "lgb_params": result.lgb_params,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
