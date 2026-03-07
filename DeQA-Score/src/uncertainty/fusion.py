"""Four-signal uncertainty fusion with tiered acceptance.

Combines four independent uncertainty signals to decide whether a SigLIP2
pseudo-label should be accepted, down-weighted, sent to Tier-2 VLM validation,
or rejected entirely.

Signals:
    1. Mahalanobis distance (d_M) — epistemic/OOD from SigLIP2 embeddings
    2. Cross-model JSD — epistemic/disagreement between SigLIP2 and DeQA
    3. SigLIP2 σ² — aleatoric uncertainty from GaussianNLL output
    4. SigLIP2 entropy — H(discretized level_probs), free signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .cross_validator import CrossValidationResult

logger = logging.getLogger(__name__)


class AcceptanceTier(str, Enum):
    """Tiered acceptance decisions for pseudo-labels."""

    AUTO_ACCEPT = "auto_accept"  # Full confidence (weight=1.0)
    LOW_WEIGHT = "low_weight"  # Accept with reduced weight (0.3-0.6)
    TIER2_TRIGGER = "tier2_trigger"  # Send to VLM for veto validation
    HARD_REJECT = "hard_reject"  # Extremely OOD — skip entirely


@dataclass(frozen=True)
class UncertaintySignals:
    """All uncertainty signals for one image+dimension."""

    mahalanobis_distance: float
    cross_model_jsd: float
    siglip2_sigma_sq: float
    siglip2_entropy: float


@dataclass(frozen=True)
class AcceptanceDecision:
    """Decision for one image+dimension with full signal trace."""

    image_id: str
    dimension: str
    tier: AcceptanceTier
    confidence_weight: float  # 0.0 (reject) to 1.0 (full accept)
    signals: UncertaintySignals
    reason: str  # Human-readable explanation


@dataclass
class JSDThresholds:
    """Per-dimension JSD thresholds, calibrated via JSD-vs-MAE analysis."""

    auto_accept: float = 0.06
    low_weight: float = 0.12


class UncertaintyFusion:
    """Four-signal uncertainty fusion with per-dimension JSD thresholds.

    Default thresholds are initial values. Call calibrate_jsd_thresholds()
    with DIQA-5000 val data to set empirical per-dimension thresholds.

    Args:
        mahalanobis_ood_threshold: d_M above which image is flagged OOD (test p95).
        mahalanobis_hard_reject: d_M above which image is hard-rejected (test p99).
        sigma_sq_auto: σ² threshold for auto-accept. Default 0.64 = 0.8²
            (matching DeQA's σ_pseudo=0.8 training default).
        sigma_sq_low: σ² threshold for low-weight acceptance.
        entropy_auto: Entropy threshold for auto-accept.
        entropy_low: Entropy threshold for low-weight acceptance.
        jsd_thresholds: Per-dimension JSD thresholds.
        low_weight_range: (min, max) confidence weight for low-weight tier.
    """

    def __init__(
        self,
        mahalanobis_ood_threshold: float = 46.0,
        mahalanobis_hard_reject: float = 58.6,
        sigma_sq_auto: float = 0.64,
        sigma_sq_low: float = 1.0,
        entropy_auto: float = 1.2,
        entropy_low: float = 1.5,
        jsd_thresholds: dict[str, JSDThresholds] | None = None,
        low_weight_range: tuple[float, float] = (0.3, 0.6),
    ) -> None:
        self.mahalanobis_ood_threshold = mahalanobis_ood_threshold
        self.mahalanobis_hard_reject = mahalanobis_hard_reject
        self.sigma_sq_auto = sigma_sq_auto
        self.sigma_sq_low = sigma_sq_low
        self.entropy_auto = entropy_auto
        self.entropy_low = entropy_low
        self.low_weight_range = low_weight_range

        # Default per-dimension JSD thresholds
        self.jsd_thresholds: dict[str, JSDThresholds] = jsd_thresholds or {
            "overall": JSDThresholds(auto_accept=0.06, low_weight=0.12),
            "sharpness": JSDThresholds(auto_accept=0.06, low_weight=0.12),
            "color": JSDThresholds(auto_accept=0.06, low_weight=0.12),
        }

    def compute_signals(
        self,
        cross_val_result: CrossValidationResult,
        mahalanobis_distance: float,
    ) -> UncertaintySignals:
        """Extract all four uncertainty signals.

        Args:
            cross_val_result: Output from CrossValidator.validate().
            mahalanobis_distance: From OODDetectorWrapper.score().
        """
        return UncertaintySignals(
            mahalanobis_distance=mahalanobis_distance,
            cross_model_jsd=cross_val_result.cross_model_jsd,
            siglip2_sigma_sq=cross_val_result.siglip2_sigma_sq,
            siglip2_entropy=cross_val_result.siglip2_entropy,
        )

    def decide(
        self,
        cross_val_result: CrossValidationResult,
        mahalanobis_distance: float,
    ) -> AcceptanceDecision:
        """Make tiered acceptance decision based on four signals.

        Decision logic (evaluated in order):
        1. Hard reject if d_M > hard_reject threshold
        2. Tier-2 trigger if d_M > OOD threshold
        3. Auto-accept if ALL signals below auto thresholds
        4. Low-weight if ALL signals below low thresholds
        5. Otherwise: tier-2 trigger

        Args:
            cross_val_result: Output from CrossValidator.validate().
            mahalanobis_distance: From OODDetectorWrapper.score().

        Returns:
            AcceptanceDecision with tier, weight, and explanation.
        """
        signals = self.compute_signals(cross_val_result, mahalanobis_distance)
        dimension = cross_val_result.dimension
        image_id = cross_val_result.image_id

        jsd_thresh = self.jsd_thresholds.get(dimension, JSDThresholds())

        # 1. Hard reject: extremely OOD
        if signals.mahalanobis_distance > self.mahalanobis_hard_reject:
            return AcceptanceDecision(
                image_id=image_id,
                dimension=dimension,
                tier=AcceptanceTier.HARD_REJECT,
                confidence_weight=0.0,
                signals=signals,
                reason=f"d_M={signals.mahalanobis_distance:.1f} > {self.mahalanobis_hard_reject} (hard reject)",
            )

        # 2. OOD trigger: send to tier-2
        if signals.mahalanobis_distance > self.mahalanobis_ood_threshold:
            return AcceptanceDecision(
                image_id=image_id,
                dimension=dimension,
                tier=AcceptanceTier.TIER2_TRIGGER,
                confidence_weight=0.0,
                signals=signals,
                reason=f"d_M={signals.mahalanobis_distance:.1f} > {self.mahalanobis_ood_threshold} (OOD)",
            )

        # 3. Auto-accept: all signals below auto thresholds
        auto_pass = (
            signals.cross_model_jsd <= jsd_thresh.auto_accept
            and signals.siglip2_sigma_sq <= self.sigma_sq_auto
            and signals.siglip2_entropy <= self.entropy_auto
        )
        if auto_pass:
            return AcceptanceDecision(
                image_id=image_id,
                dimension=dimension,
                tier=AcceptanceTier.AUTO_ACCEPT,
                confidence_weight=1.0,
                signals=signals,
                reason="All signals within auto-accept thresholds",
            )

        # 4. Low-weight: all signals below low thresholds
        low_pass = (
            signals.cross_model_jsd <= jsd_thresh.low_weight
            and signals.siglip2_sigma_sq <= self.sigma_sq_low
            and signals.siglip2_entropy <= self.entropy_low
        )
        if low_pass:
            # Interpolate weight based on how far signals are from auto thresholds
            weight = self._interpolate_weight(signals, jsd_thresh)
            return AcceptanceDecision(
                image_id=image_id,
                dimension=dimension,
                tier=AcceptanceTier.LOW_WEIGHT,
                confidence_weight=weight,
                signals=signals,
                reason=f"Signals within low-weight range (weight={weight:.2f})",
            )

        # 5. Tier-2 trigger: signals exceed low thresholds
        reasons = []
        if signals.cross_model_jsd > jsd_thresh.low_weight:
            reasons.append(f"JSD={signals.cross_model_jsd:.3f}")
        if signals.siglip2_sigma_sq > self.sigma_sq_low:
            reasons.append(f"σ²={signals.siglip2_sigma_sq:.3f}")
        if signals.siglip2_entropy > self.entropy_low:
            reasons.append(f"H={signals.siglip2_entropy:.3f}")

        return AcceptanceDecision(
            image_id=image_id,
            dimension=dimension,
            tier=AcceptanceTier.TIER2_TRIGGER,
            confidence_weight=0.0,
            signals=signals,
            reason=f"Exceeded low-weight thresholds: {', '.join(reasons)}",
        )

    def _interpolate_weight(
        self,
        signals: UncertaintySignals,
        jsd_thresh: JSDThresholds,
    ) -> float:
        """Interpolate confidence weight in the low-weight range.

        Uses the worst (most uncertain) signal relative to its auto/low range
        to determine the weight.
        """
        lo, hi = self.low_weight_range

        # Compute normalized position for each signal (0=auto threshold, 1=low threshold)
        jsd_range = jsd_thresh.low_weight - jsd_thresh.auto_accept
        sigma_range = self.sigma_sq_low - self.sigma_sq_auto
        entropy_range = self.entropy_low - self.entropy_auto

        positions = []
        if jsd_range > 0:
            jsd_pos = (signals.cross_model_jsd - jsd_thresh.auto_accept) / jsd_range
            positions.append(np.clip(jsd_pos, 0.0, 1.0))
        if sigma_range > 0:
            sigma_pos = (signals.siglip2_sigma_sq - self.sigma_sq_auto) / sigma_range
            positions.append(np.clip(sigma_pos, 0.0, 1.0))
        if entropy_range > 0:
            entropy_pos = (signals.siglip2_entropy - self.entropy_auto) / entropy_range
            positions.append(np.clip(entropy_pos, 0.0, 1.0))

        if not positions:
            return hi

        # Use worst (highest) signal position
        worst = max(positions)
        # Higher uncertainty → lower weight
        return float(hi - worst * (hi - lo))

    def calibrate_jsd_thresholds(
        self,
        val_results: list[CrossValidationResult],
        ground_truth_mos: dict[str, dict[str, float]],
        mae_auto_threshold: float = 0.5,
        mae_low_threshold: float = 0.75,
    ) -> dict[str, JSDThresholds]:
        """Calibrate per-dimension JSD thresholds using JSD-vs-MAE analysis.

        For each dimension, bins results by JSD deciles and finds the JSD value
        where mean MAE exceeds the given thresholds.

        Args:
            val_results: CrossValidationResults on DIQA-5000 val set.
            ground_truth_mos: {dimension: {image_id: mos_score}}.
            mae_auto_threshold: MAE above which auto-accept is too risky.
            mae_low_threshold: MAE above which low-weight is too risky.

        Returns:
            Dict of per-dimension JSDThresholds.
        """
        calibrated: dict[str, JSDThresholds] = {}

        for dimension in ["overall", "sharpness", "color"]:
            dim_results = [r for r in val_results if r.dimension == dimension]
            if not dim_results or dimension not in ground_truth_mos:
                calibrated[dimension] = JSDThresholds()
                continue

            gt = ground_truth_mos[dimension]
            jsds = []
            maes = []
            for r in dim_results:
                if r.image_id not in gt:
                    continue
                jsds.append(r.cross_model_jsd)
                maes.append(abs(r.siglip2_mu - gt[r.image_id]))

            if len(jsds) < 20:
                calibrated[dimension] = JSDThresholds()
                continue

            jsds_arr = np.array(jsds)
            maes_arr = np.array(maes)

            # Sort by JSD
            order = np.argsort(jsds_arr)
            jsds_sorted = jsds_arr[order]
            maes_sorted = maes_arr[order]

            # Bin into deciles
            n_bins = 10
            bin_size = len(jsds_sorted) // n_bins

            auto_jsd = jsds_sorted[-1]  # default: max JSD (accept all)
            low_jsd = jsds_sorted[-1]

            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size if i < n_bins - 1 else len(jsds_sorted)
                bin_mae = maes_sorted[start:end].mean()
                jsds_sorted[end - 1]

                if bin_mae > mae_auto_threshold and auto_jsd == jsds_sorted[-1]:
                    auto_jsd = jsds_sorted[max(0, start - 1)]
                if bin_mae > mae_low_threshold and low_jsd == jsds_sorted[-1]:
                    low_jsd = jsds_sorted[max(0, start - 1)]

            calibrated[dimension] = JSDThresholds(
                auto_accept=float(auto_jsd),
                low_weight=float(low_jsd),
            )
            logger.info(
                "Calibrated %s JSD thresholds: auto=%.4f, low=%.4f",
                dimension,
                auto_jsd,
                low_jsd,
            )

        self.jsd_thresholds = calibrated
        return calibrated
