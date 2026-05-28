"""Per-model z-score normalization fitted on in-distribution data.

Fits mean/std per model on DIQA-5000 predictions, then normalizes new
predictions to a common scale. Critical: normalization parameters MUST
come from in-distribution data only, never fit on OOD data.

Serialization follows the OODDetectorWrapper.from_npz() pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationParams:
    """Per-model normalization parameters."""

    model_name: str
    mean: float
    std: float


class ModelNormalizer:
    """Z-score normalizer fitted on in-distribution model predictions.

    Fits per-model mean/std on DIQA-5000, then applies the same transform
    to any new predictions. This preserves the spread signal, if models
    agree on ID data but disagree on OOD data, normalized spread will be
    larger for OOD.

    Args:
        params: Pre-fitted normalization parameters. If None, must call fit().
    """

    def __init__(self, params: dict[str, NormalizationParams] | None = None) -> None:
        self._params: dict[str, NormalizationParams] = params or {}

    @property
    def model_names(self) -> list[str]:
        """Names of all fitted models."""
        return list(self._params.keys())

    @property
    def is_fitted(self) -> bool:
        """Whether normalization parameters have been fitted."""
        return len(self._params) > 0

    def get_params(self, model_name: str) -> NormalizationParams:
        """Get normalization parameters for a specific model.

        Raises:
            KeyError: If model_name has not been fitted.
        """
        if model_name not in self._params:
            msg = f"Model '{model_name}' not fitted. Available: {self.model_names}"
            raise KeyError(msg)
        return self._params[model_name]

    def fit(self, predictions: dict[str, list[float]]) -> None:
        """Fit normalization parameters from in-distribution predictions.

        Args:
            predictions: {model_name: [score1, score2, ...]} from DIQA-5000.
                Each list must have the same length (one score per image).

        Raises:
            ValueError: If any model has fewer than 10 predictions or std=0.
        """
        self._params = {}
        for model_name, scores in predictions.items():
            arr = np.asarray(scores, dtype=np.float64)
            if len(arr) < 10:
                msg = f"Model '{model_name}' has {len(arr)} predictions, need >= 10"
                raise ValueError(msg)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            if std < 1e-10:
                msg = f"Model '{model_name}' has zero variance (std={std})"
                raise ValueError(msg)
            self._params[model_name] = NormalizationParams(
                model_name=model_name, mean=mean, std=std
            )
            logger.debug(
                "Fitted %s: mean=%.4f, std=%.4f (n=%d)",
                model_name,
                mean,
                std,
                len(arr),
            )

    def transform(self, model_name: str, score: float) -> float:
        """Z-score normalize a single prediction.

        Args:
            model_name: Which model produced the score.
            score: Raw model prediction.

        Returns:
            Normalized score: (score - mean) / std.

        Raises:
            KeyError: If model_name has not been fitted.
        """
        params = self.get_params(model_name)
        return (score - params.mean) / params.std

    def transform_batch(self, predictions: dict[str, float]) -> dict[str, float]:
        """Normalize predictions from multiple models for one image.

        Args:
            predictions: {model_name: raw_score} for a single image.

        Returns:
            {model_name: normalized_score} for fitted models only.
            Models not in predictions are silently skipped.

        Raises:
            KeyError: If a model in predictions has not been fitted.
        """
        return {
            name: self.transform(name, score)
            for name, score in predictions.items()
            if name in self._params
        }

    def to_npz(self, path: str | Path) -> None:
        """Save normalization parameters to .npz file.

        File contains arrays: model_names, means, stds.
        """
        if not self._params:
            msg = "Cannot save unfitted normalizer"
            raise ValueError(msg)

        names = list(self._params.keys())
        means = np.array([self._params[n].mean for n in names])
        stds = np.array([self._params[n].std for n in names])

        np.savez(
            path,
            model_names=np.array(names),
            means=means,
            stds=stds,
        )
        logger.info("Saved normalizer with %d models to %s", len(names), path)

    @classmethod
    def from_npz(cls, path: str | Path) -> ModelNormalizer:
        """Load normalization parameters from .npz file.

        Args:
            path: Path to .npz file saved by to_npz().

        Returns:
            ModelNormalizer with pre-fitted parameters.
        """
        data = np.load(path, allow_pickle=False)
        names = data["model_names"]
        means = data["means"]
        stds = data["stds"]

        params = {}
        for name, mean, std in zip(names, means, stds):
            name_str = str(name)
            params[name_str] = NormalizationParams(
                model_name=name_str,
                mean=float(mean),
                std=float(std),
            )

        return cls(params=params)
