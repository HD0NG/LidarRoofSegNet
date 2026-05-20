"""Merge scorers.

The merge classifier is the trainable component (ADR-0002). Three concrete
implementations live here:

* :class:`HandTunedScorer` — the natural ablation: a transparent geometric
  scoring rule on the same features. Required by ADR-0002 as the baseline
  the learned classifier must beat.
* :class:`LightGBMScorer` — the headline configuration. Loads a saved
  LightGBM model; raises a clear error if lightgbm isn't installed.
* :class:`MergeScorer` Protocol — interface every scorer satisfies.

Until the LightGBM classifier is trained (separate roadmap step), use
``HandTunedScorer`` as the default in the pipeline.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from roofseg.refinement.features import FEATURE_ORDER, PairFeatures


@runtime_checkable
class MergeScorer(Protocol):
    """Score a candidate merge: higher means more likely to be one face."""

    name: str

    def score(self, features: PairFeatures) -> float: ...

    def score_batch(self, batch: Iterable[PairFeatures]) -> np.ndarray:
        ...


class HandTunedScorer:
    """Transparent geometric scorer used as the ablation baseline.

    The score is a soft conjunction of four signals:

    * **Coplanarity**: small normal angle.
    * **Joint-fit quality**: low joint plane residual.
    * **Proximity**: small boundary distance.
    * **Offset agreement**: matching plane offsets.

    Each signal is mapped to a sigmoid-like score in [0, 1] and combined as
    a geometric mean. The thresholds below match the scale the roofNTNU
    pipeline normalises to (unit-sphere); pass overrides if you change the
    normalisation.
    """

    name = "hand_tuned"

    def __init__(
        self,
        *,
        normal_angle_tol: float = 0.20,  # radians, ≈ 11.5°
        joint_residual_tol: float = 0.02,  # in normalised units
        boundary_distance_tol: float = 0.05,
        offset_tol: float = 0.05,
    ):
        self.normal_angle_tol = normal_angle_tol
        self.joint_residual_tol = joint_residual_tol
        self.boundary_distance_tol = boundary_distance_tol
        self.offset_tol = offset_tol

    @staticmethod
    def _soft(value: float, tol: float) -> float:
        if tol <= 0:
            return 0.0 if value > 0 else 1.0
        # Smooth step: 1 at value=0, 0.5 at value=tol, ≈0 beyond ~2·tol.
        return float(1.0 / (1.0 + (value / tol) ** 2))

    def score(self, features: PairFeatures) -> float:
        s_normal = self._soft(features.normal_angle, self.normal_angle_tol)
        s_joint = self._soft(features.joint_residual, self.joint_residual_tol)
        s_boundary = self._soft(features.boundary_distance, self.boundary_distance_tol)
        s_offset = self._soft(features.offset_diff, self.offset_tol)
        # Geometric mean — any single signal can veto the merge.
        product = max(s_normal * s_joint * s_boundary * s_offset, 1e-12)
        return float(product ** 0.25)

    def score_batch(self, batch: Iterable[PairFeatures]) -> np.ndarray:
        return np.array([self.score(f) for f in batch], dtype=np.float64)


class LightGBMScorer:
    """Wraps a trained LightGBM model that returns ``P(should_merge)``.

    The classifier is trained on cluster-pair labels harvested via k-fold
    cross-prediction (ADR-0002 + roadmap step 4). Feature order is fixed by
    :data:`roofseg.refinement.features.FEATURE_ORDER`.
    """

    name = "lightgbm"

    def __init__(self, model_path: str | None = None, *, booster=None):
        if booster is None and model_path is None:
            raise ValueError("provide either model_path or booster=")
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError as e:  # pragma: no cover - exercised in CI without lightgbm
            raise RuntimeError(
                "LightGBMScorer needs lightgbm — install with `pip install lightgbm` "
                "or `pip install -e .[classifier]`"
            ) from e

        import lightgbm as lgb

        if booster is not None:
            self.booster = booster
        else:
            self.booster = lgb.Booster(model_file=model_path)

    def score(self, features: PairFeatures) -> float:
        return float(self.score_batch([features])[0])

    def score_batch(self, batch: Iterable[PairFeatures]) -> np.ndarray:
        rows = np.stack([f.to_vector() for f in batch], axis=0)
        return self.booster.predict(rows).astype(np.float64)


def default_scorer() -> MergeScorer:
    """The scorer the pipeline uses when nothing is passed in.

    Currently :class:`HandTunedScorer`; will become :class:`LightGBMScorer`
    once the merge classifier is trained and packaged with the release.
    """
    return HandTunedScorer()


def features_to_matrix(batch: Iterable[PairFeatures]) -> np.ndarray:
    """Stack PairFeatures into an ``(N, F)`` matrix in :data:`FEATURE_ORDER`."""
    return np.stack([f.to_vector() for f in batch], axis=0)


def feature_names() -> list[str]:
    return list(FEATURE_ORDER)
