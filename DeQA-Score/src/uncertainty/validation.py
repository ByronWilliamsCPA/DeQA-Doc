"""Validation and monitoring for pseudo-label quality.

Provides bootstrap confidence intervals, harm checks against baseline,
and distribution drift monitoring to ensure pseudo-labels don't degrade
model performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .gaussian_to_discrete import level_probs_to_mos

logger = logging.getLogger(__name__)

# Minimum acceptable SRCC on sacred test set
SRCC_FLOOR = 0.65


def _srcc(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    d = rx - ry
    return float(1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1)))


def _plcc(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson linear correlation coefficient."""
    if len(x) < 3:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    dx, dy = x - mx, y - my
    num = np.sum(dx * dy)
    denom = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))
    if denom < 1e-15:
        return 0.0
    return float(num / denom)


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""

    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstrap: int


@dataclass(frozen=True)
class HarmCheckResult:
    """Result of comparing current model against baseline."""

    dimension: str
    baseline_srcc: float
    current_srcc: float
    srcc_delta: float
    is_harmful: bool
    tolerance: float
    baseline_plcc: float
    current_plcc: float


class PseudoLabelValidator:
    """Validates pseudo-label quality and monitors for performance regression.

    Args:
        srcc_floor: Minimum acceptable SRCC on the sacred test set.
        harm_tolerance: Maximum acceptable SRCC drop vs baseline.
    """

    def __init__(
        self,
        srcc_floor: float = SRCC_FLOOR,
        harm_tolerance: float = 0.02,
    ) -> None:
        self.srcc_floor = srcc_floor
        self.harm_tolerance = harm_tolerance

    def bootstrap_ci(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metric_fn: str = "srcc",
        n_bootstrap: int = 1000,
        alpha: float = 0.95,
        seed: int = 42,
    ) -> BootstrapCI:
        """Compute bootstrap confidence interval for a correlation metric.

        Args:
            predictions: Model predictions, shape (N,).
            ground_truth: Ground truth values, shape (N,).
            metric_fn: "srcc" or "plcc".
            n_bootstrap: Number of bootstrap resamples.
            alpha: Confidence level (0.95 = 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            BootstrapCI with point estimate and confidence bounds.
        """
        fn = _srcc if metric_fn == "srcc" else _plcc
        predictions = np.asarray(predictions, dtype=np.float64)
        ground_truth = np.asarray(ground_truth, dtype=np.float64)

        point_estimate = fn(predictions, ground_truth)

        rng = np.random.default_rng(seed)
        n = len(predictions)
        bootstrap_values = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bootstrap_values[i] = fn(predictions[idx], ground_truth[idx])

        tail = (1.0 - alpha) / 2.0
        ci_lower = float(np.percentile(bootstrap_values, 100.0 * tail))
        ci_upper = float(np.percentile(bootstrap_values, 100.0 * (1.0 - tail)))

        return BootstrapCI(
            metric_name=metric_fn,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
        )

    def harm_check(
        self,
        baseline_preds: np.ndarray,
        current_preds: np.ndarray,
        ground_truth: np.ndarray,
        dimension: str = "overall",
    ) -> HarmCheckResult:
        """Check if current model is worse than baseline beyond tolerance.

        Args:
            baseline_preds: Predictions from baseline model.
            current_preds: Predictions from model trained with pseudo-labels.
            ground_truth: Ground truth scores.
            dimension: Quality dimension being evaluated.

        Returns:
            HarmCheckResult indicating whether regression occurred.
        """
        baseline_srcc = _srcc(np.asarray(baseline_preds), np.asarray(ground_truth))
        current_srcc = _srcc(np.asarray(current_preds), np.asarray(ground_truth))
        baseline_plcc = _plcc(np.asarray(baseline_preds), np.asarray(ground_truth))
        current_plcc = _plcc(np.asarray(current_preds), np.asarray(ground_truth))

        delta = current_srcc - baseline_srcc
        is_harmful = delta < -self.harm_tolerance

        if is_harmful:
            logger.warning(
                "HARM CHECK FAILED for %s: SRCC dropped %.4f (%.4f → %.4f), tolerance=%.4f",
                dimension,
                abs(delta),
                baseline_srcc,
                current_srcc,
                self.harm_tolerance,
            )

        return HarmCheckResult(
            dimension=dimension,
            baseline_srcc=baseline_srcc,
            current_srcc=current_srcc,
            srcc_delta=delta,
            is_harmful=is_harmful,
            tolerance=self.harm_tolerance,
            baseline_plcc=baseline_plcc,
            current_plcc=current_plcc,
        )

    def check_srcc_floor(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        dimension: str = "overall",
    ) -> bool:
        """Check if SRCC meets minimum floor on sacred test set.

        Args:
            predictions: Model predictions on sacred test set.
            ground_truth: Ground truth for sacred test set.
            dimension: Quality dimension.

        Returns:
            True if SRCC >= floor, False otherwise.
        """
        srcc = _srcc(np.asarray(predictions), np.asarray(ground_truth))
        passes = srcc >= self.srcc_floor

        if not passes:
            logger.warning(
                "SRCC FLOOR CHECK FAILED for %s: %.4f < %.4f",
                dimension,
                srcc,
                self.srcc_floor,
            )
        else:
            logger.info(
                "SRCC floor check passed for %s: %.4f >= %.4f",
                dimension,
                srcc,
                self.srcc_floor,
            )

        return passes

    @staticmethod
    def distribution_drift(
        pseudo_probs: list[np.ndarray],
        training_probs: list[np.ndarray],
    ) -> dict[str, float]:
        """Compare pseudo-label distribution against training distribution.

        Detects if pseudo-labels have significantly different quality
        distribution than the original training data (potential bias).

        Args:
            pseudo_probs: List of level_probs arrays from pseudo-labels.
            training_probs: List of level_probs arrays from training data.

        Returns:
            Dict with drift metrics: mean_mos_delta, mos_ks_statistic,
            per_level_mean_delta.
        """
        pseudo_mos = np.array([level_probs_to_mos(p) for p in pseudo_probs])
        training_mos = np.array([level_probs_to_mos(p) for p in training_probs])

        mean_mos_delta = float(abs(np.mean(pseudo_mos) - np.mean(training_mos)))

        # KS statistic for MOS distributions
        combined = np.concatenate([pseudo_mos, training_mos])
        sorted_vals = np.sort(np.unique(combined))
        max_diff = 0.0
        for val in sorted_vals:
            ecdf_pseudo = np.mean(pseudo_mos <= val)
            ecdf_training = np.mean(training_mos <= val)
            max_diff = max(max_diff, abs(ecdf_pseudo - ecdf_training))

        # Per-level mean probability comparison
        pseudo_mean = np.mean(pseudo_probs, axis=0)
        training_mean = np.mean(training_probs, axis=0)
        per_level_delta = np.abs(pseudo_mean - training_mean)

        return {
            "mean_mos_delta": mean_mos_delta,
            "mos_ks_statistic": float(max_diff),
            "per_level_mean_delta_excellent": float(per_level_delta[0]),
            "per_level_mean_delta_good": float(per_level_delta[1]),
            "per_level_mean_delta_fair": float(per_level_delta[2]),
            "per_level_mean_delta_poor": float(per_level_delta[3]),
            "per_level_mean_delta_bad": float(per_level_delta[4]),
        }

    @staticmethod
    def per_category_acceptance_audit(
        samples: list[dict],
        category_key: str = "category",
    ) -> dict[str, dict[str, float]]:
        """Audit acceptance rates per category to detect shared blind spots.

        Categories where >95% of samples are auto-accepted may indicate
        both SigLIP2 and DeQA agree but are both wrong.

        Args:
            samples: List of dicts with 'category', 'tier', etc.
            category_key: Key in sample dicts for category grouping.

        Returns:
            Dict[category] → {total, auto_accept_rate, mean_weight}.
        """
        by_category: dict[str, list[dict]] = {}
        for s in samples:
            cat = s.get(category_key, "unknown")
            by_category.setdefault(cat, []).append(s)

        audit: dict[str, dict[str, float]] = {}
        for cat, cat_samples in by_category.items():
            total = len(cat_samples)
            auto_count = sum(1 for s in cat_samples if s.get("tier") == "auto_accept")
            weights = [s.get("confidence_weight", 0.0) for s in cat_samples]

            rate = auto_count / total if total > 0 else 0.0
            audit[cat] = {
                "total": float(total),
                "auto_accept_rate": round(rate, 4),
                "mean_weight": round(float(np.mean(weights)) if weights else 0.0, 4),
            }

            if rate > 0.95 and total >= 10:
                logger.warning(
                    "BLIND SPOT WARNING: category '%s' has %.1f%% auto-accept rate "
                    "(%d samples) — may indicate shared model blind spot",
                    cat,
                    rate * 100,
                    total,
                )

        return audit
