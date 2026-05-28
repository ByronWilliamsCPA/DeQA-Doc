"""Multi-model spread computation for OOD detection.

Computes per-image spread (standard deviation of z-score-normalized
predictions across architecturally diverse models) as an epistemic
uncertainty signal. High spread indicates models disagree, a strong
proxy for out-of-distribution documents.

Research basis: research/correlation/ood_spread_analysis.py demonstrated
2.21x spread amplification on OOD vs in-distribution data, with spread
uncorrelated to prediction error on ID data (pure OOD signal).

Ensemble: SigLIP2 (vision regression) + DeQA specialist (mPLUG-Owl2
generative) + Qwen2.5-VL-7B (Qwen generative). Architectural diversity
is critical, homogeneous models produce weak spread signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model_normalizer import ModelNormalizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpreadConfig:
    """Configuration for spread computation.

    Args:
        model_names: Models participating in spread computation.
        vision_models: Subset of model_names that are vision-only (no LLM decoder).
        mllm_models: Subset of model_names that are multimodal LLMs.
        min_models: Minimum models required for spread computation.
            Falls back gracefully when fewer predictions available.
    """

    model_names: tuple[str, ...] = ("siglip2", "deqa_specialist", "qwen25vl")
    vision_models: tuple[str, ...] = ("siglip2",)
    mllm_models: tuple[str, ...] = ("deqa_specialist", "qwen25vl")
    min_models: int = 2


@dataclass(frozen=True)
class BaselineSpreadStats:
    """Spread distribution statistics from in-distribution (DIQA-5000) data.

    Thresholds are percentile-based (not sigma-multipliers) because
    n=3 model SD follows a non-Gaussian distribution (chi-squared, df=2).

    Args:
        mean: Mean spread across DIQA-5000 images.
        std: Standard deviation of spread distribution.
        median: Median spread.
        q75: 75th percentile, candidate for auto-accept threshold.
        q90: 90th percentile, soft OOD threshold.
        q95: 95th percentile.
        q99: 99th percentile, hard OOD threshold.
        soft_threshold: Spread above which image is soft OOD (default: q90).
        hard_threshold: Spread above which image is strong OOD (default: q99).
        n_images: Number of images used to compute baseline.
    """

    mean: float
    std: float
    median: float
    q75: float
    q90: float
    q95: float
    q99: float
    soft_threshold: float
    hard_threshold: float
    n_images: int

    def to_npz(self, path: str | Path) -> None:
        """Save baseline stats to .npz file."""
        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            median=self.median,
            q75=self.q75,
            q90=self.q90,
            q95=self.q95,
            q99=self.q99,
            soft_threshold=self.soft_threshold,
            hard_threshold=self.hard_threshold,
            n_images=self.n_images,
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> BaselineSpreadStats:
        """Load baseline stats from .npz file."""
        data = np.load(path, allow_pickle=False)
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            median=float(data["median"]),
            q75=float(data["q75"]),
            q90=float(data["q90"]),
            q95=float(data["q95"]),
            q99=float(data["q99"]),
            soft_threshold=float(data["soft_threshold"]),
            hard_threshold=float(data["hard_threshold"]),
            n_images=int(data["n_images"]),
        )


@dataclass(frozen=True)
class SpreadResult:
    """Spread computation result for a single image.

    Args:
        spread: Standard deviation across normalized model predictions.
        cluster_divergence: |mean(vision_models) - mean(mllm_models)|.
            Measures architectural disagreement between model families.
        ood_category: Classification against baseline thresholds.
            0 = in-distribution-like, 1 = soft OOD, 2 = strong OOD.
        normalized_scores: {model_name: z-score} for debugging/provenance.
        n_models_used: Number of models that contributed to this spread.
    """

    spread: float
    cluster_divergence: float
    ood_category: int
    normalized_scores: dict[str, float]
    n_models_used: int


class SpreadComputer:
    """Compute inter-model spread as an OOD detection signal.

    Requires a pre-fitted ModelNormalizer (fitted on DIQA-5000) and
    BaselineSpreadStats for threshold-based OOD classification.

    Args:
        normalizer: Pre-fitted ModelNormalizer for z-score normalization.
        baseline_stats: Spread distribution from DIQA-5000 baseline.
        config: Spread computation configuration.
    """

    def __init__(
        self,
        normalizer: ModelNormalizer,
        baseline_stats: BaselineSpreadStats,
        config: SpreadConfig | None = None,
    ) -> None:
        self.normalizer = normalizer
        self.baseline_stats = baseline_stats
        self.config = config or SpreadConfig()

    def compute(self, predictions: dict[str, float]) -> SpreadResult:
        """Compute spread for a single image from multiple model predictions.

        Args:
            predictions: {model_name: raw_score} for one image.
                Only models present in both predictions and the normalizer
                are used. Must have at least config.min_models.

        Returns:
            SpreadResult with spread, cluster divergence, and OOD category.

        Raises:
            ValueError: If fewer than min_models predictions are available.
        """
        # Normalize only models we have both predictions and params for
        available = {
            name: score
            for name, score in predictions.items()
            if name in self.normalizer.model_names
        }

        if len(available) < self.config.min_models:
            msg = (
                f"Need >= {self.config.min_models} model predictions, "
                f"got {len(available)}: {list(available.keys())}"
            )
            raise ValueError(msg)

        normalized = self.normalizer.transform_batch(available)
        scores_array = np.array(list(normalized.values()), dtype=np.float64)

        # Spread = SD across normalized predictions
        spread = float(scores_array.std(ddof=0))

        # Cluster divergence = |mean(vision) - mean(mllm)|
        cluster_divergence = self._compute_cluster_divergence(normalized)

        # OOD classification against baseline thresholds
        ood_category = self._classify_ood(spread)

        return SpreadResult(
            spread=spread,
            cluster_divergence=cluster_divergence,
            ood_category=ood_category,
            normalized_scores=dict(normalized),
            n_models_used=len(normalized),
        )

    def compute_batch(
        self, batch_predictions: list[dict[str, float]]
    ) -> list[SpreadResult]:
        """Compute spread for a batch of images.

        Args:
            batch_predictions: List of {model_name: raw_score} per image.

        Returns:
            List of SpreadResult, one per image.
        """
        return [self.compute(preds) for preds in batch_predictions]

    def _compute_cluster_divergence(self, normalized: dict[str, float]) -> float:
        """Compute |mean(vision) - mean(mllm)| from normalized scores."""
        vision_scores = [
            normalized[m] for m in self.config.vision_models if m in normalized
        ]
        mllm_scores = [
            normalized[m] for m in self.config.mllm_models if m in normalized
        ]

        if not vision_scores or not mllm_scores:
            return 0.0

        vision_mean = float(np.mean(vision_scores))
        mllm_mean = float(np.mean(mllm_scores))
        return abs(vision_mean - mllm_mean)

    def _classify_ood(self, spread: float) -> int:
        """Classify spread against baseline thresholds.

        Returns:
            0 = in-distribution-like (spread <= soft_threshold)
            1 = soft OOD (soft_threshold < spread <= hard_threshold)
            2 = strong OOD (spread > hard_threshold)
        """
        if spread > self.baseline_stats.hard_threshold:
            return 2
        if spread > self.baseline_stats.soft_threshold:
            return 1
        return 0

    @classmethod
    def fit_baseline(
        cls,
        normalizer: ModelNormalizer,
        diqa_predictions: list[dict[str, float]],
        config: SpreadConfig | None = None,
    ) -> BaselineSpreadStats:
        """Compute baseline spread statistics from DIQA-5000 predictions.

        This is the calibration step, run once offline on in-distribution
        data to establish the spread distribution and thresholds.

        Args:
            normalizer: Pre-fitted ModelNormalizer.
            diqa_predictions: List of {model_name: raw_score} for each
                DIQA-5000 image. All config.model_names should be present.
            config: Spread configuration.

        Returns:
            BaselineSpreadStats with percentile-based thresholds.

        Raises:
            ValueError: If fewer than 50 images have sufficient predictions.
        """
        config = config or SpreadConfig()

        spreads = []
        for preds in diqa_predictions:
            available = {
                name: score
                for name, score in preds.items()
                if name in normalizer.model_names
            }
            if len(available) < config.min_models:
                continue

            normalized = normalizer.transform_batch(available)
            scores_array = np.array(list(normalized.values()), dtype=np.float64)
            spreads.append(float(scores_array.std(ddof=0)))

        spreads_arr = np.array(spreads, dtype=np.float64)
        if len(spreads_arr) < 50:
            msg = f"Need >= 50 valid predictions for baseline, got {len(spreads_arr)}"
            raise ValueError(msg)

        q75 = float(np.percentile(spreads_arr, 75))
        q90 = float(np.percentile(spreads_arr, 90))
        q95 = float(np.percentile(spreads_arr, 95))
        q99 = float(np.percentile(spreads_arr, 99))

        stats = BaselineSpreadStats(
            mean=float(spreads_arr.mean()),
            std=float(spreads_arr.std(ddof=0)),
            median=float(np.median(spreads_arr)),
            q75=q75,
            q90=q90,
            q95=q95,
            q99=q99,
            soft_threshold=q90,
            hard_threshold=q99,
            n_images=len(spreads_arr),
        )

        logger.info(
            "Baseline spread: mean=%.4f, std=%.4f, soft=%.4f (p90), hard=%.4f (p99), n=%d",
            stats.mean,
            stats.std,
            stats.soft_threshold,
            stats.hard_threshold,
            stats.n_images,
        )

        return stats
