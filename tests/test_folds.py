"""Tests for the k-fold split manifest."""

from __future__ import annotations

import json

import pytest

from roofseg.refinement.folds import load_folds, make_folds, save_folds


def test_make_folds_is_deterministic():
    scenes = [f"scene_{i:03d}" for i in range(50)]
    a = make_folds(scenes, n_folds=5, seed=42)
    b = make_folds(scenes, n_folds=5, seed=42)
    assert a.folds == b.folds


def test_make_folds_independent_of_input_order():
    scenes = [f"scene_{i:03d}" for i in range(50)]
    a = make_folds(scenes, n_folds=5, seed=42)
    b = make_folds(list(reversed(scenes)), n_folds=5, seed=42)
    assert a.folds == b.folds


def test_make_folds_coverage_and_disjointness():
    scenes = [f"scene_{i:03d}" for i in range(47)]
    manifest = make_folds(scenes, n_folds=5, seed=42)
    seen: list[str] = []
    for ids in manifest.folds.values():
        seen.extend(ids)
    assert sorted(seen) == sorted(scenes)
    assert len(seen) == len(set(seen))


def test_make_folds_balanced_sizes():
    scenes = [f"scene_{i:03d}" for i in range(50)]
    manifest = make_folds(scenes, n_folds=5, seed=42)
    sizes = sorted(len(v) for v in manifest.folds.values())
    assert max(sizes) - min(sizes) <= 1


def test_train_scenes_excluding_fold():
    scenes = [f"scene_{i:03d}" for i in range(50)]
    manifest = make_folds(scenes, n_folds=5, seed=42)
    train = manifest.train_scenes_excluding_fold(2)
    heldout = manifest.scenes_for_fold(2)
    assert set(train).isdisjoint(set(heldout))
    assert set(train) | set(heldout) == set(scenes)


def test_save_and_load_round_trip(tmp_path):
    scenes = [f"scene_{i:03d}" for i in range(10)]
    manifest = make_folds(scenes, n_folds=3, seed=7)
    out = tmp_path / "folds.json"
    save_folds(manifest, out)

    loaded = load_folds(out)
    assert loaded.folds == manifest.folds
    assert loaded.n_folds == manifest.n_folds
    assert loaded.seed == manifest.seed
    # On-disk fold keys are stringified — JSON requirement.
    with open(out) as f:
        raw = json.load(f)
    assert set(raw["folds"].keys()) == {"0", "1", "2"}


def test_make_folds_rejects_small_n():
    with pytest.raises(ValueError):
        make_folds(["s_0", "s_1"], n_folds=1)


def test_make_folds_rejects_empty():
    with pytest.raises(ValueError):
        make_folds([], n_folds=5)
