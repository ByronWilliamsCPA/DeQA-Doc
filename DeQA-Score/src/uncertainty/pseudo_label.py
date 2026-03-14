"""End-to-end pseudo-labeling pipeline.

Orchestrates: SigLIP2 prediction → OOD check → cross-validation →
uncertainty fusion → optional VLM veto → pseudo-label output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .cross_validator import CrossValidator
from .fusion import AcceptanceDecision, AcceptanceTier, UncertaintyFusion
from .gaussian_to_discrete import (
    level_probs_to_mos,
    level_probs_to_std,
    siglip2_output_to_level_probs,
)
from .ood_wrapper import OODDetectorWrapper
from .spread import SpreadComputer, SpreadResult
from .vlm_validator import VLMValidator

logger = logging.getLogger(__name__)

DIMENSIONS = ("overall", "sharpness", "color")


@dataclass
class PseudoLabelSample:
    """A single pseudo-labeled sample with all metadata."""

    image_id: str
    dimension: str
    level_probs: np.ndarray  # (5,) [excellent→bad]
    mos: float  # Reconstructed MOS from level_probs
    std: float  # Std from level_probs distribution
    confidence_weight: float  # 0.0-1.0
    tier: AcceptanceTier
    decision: AcceptanceDecision
    vlm_vetoed: bool = False


class PseudoLabelPipeline:
    """Orchestrates the full pseudo-labeling pipeline.

    Args:
        ood_detector: Mahalanobis OOD scorer.
        cross_validator: SigLIP2 vs DeQA cross-validator.
        fusion: Uncertainty fusion engine.
        vlm_validator: Optional Tier-2 VLM validator.
        image_root: Root directory for image paths (for VLM validation).
    """

    def __init__(
        self,
        ood_detector: OODDetectorWrapper,
        cross_validator: CrossValidator,
        fusion: UncertaintyFusion,
        vlm_validator: VLMValidator | None = None,
        spread_computer: SpreadComputer | None = None,
        image_root: str = "",
    ) -> None:
        self.ood_detector = ood_detector
        self.cross_validator = cross_validator
        self.fusion = fusion
        self.vlm_validator = vlm_validator
        self.spread_computer = spread_computer
        self.image_root = image_root

    def process_single(
        self,
        image_id: str,
        siglip2_mu: float,
        siglip2_sigma_sq: float,
        embedding: np.ndarray,
        dimension: str,
        spread_result: SpreadResult | None = None,
    ) -> PseudoLabelSample:
        """Process a single image+dimension through the pipeline.

        Steps:
            1. Convert SigLIP2 (μ, σ²) → level_probs
            2. Score OOD via Mahalanobis
            3. Cross-validate against DeQA predictions
            4. Fuse signals → acceptance decision (with optional spread)
            5. Return PseudoLabelSample (VLM veto handled in process_batch)

        Args:
            image_id: Image path or identifier.
            siglip2_mu: SigLIP2 predicted mean quality.
            siglip2_sigma_sq: SigLIP2 predicted variance.
            embedding: SigLIP2 768-dim embedding for OOD scoring.
            dimension: Quality dimension.
            spread_result: Optional pre-computed spread from SpreadComputer.
                Spread is per-image (shared across dimensions), so it should
                be computed once per image and passed to all dimension calls.

        Returns:
            PseudoLabelSample with level_probs, weight, and tier.
        """
        # 1. Convert to level_probs
        level_probs = siglip2_output_to_level_probs(siglip2_mu, siglip2_sigma_sq)

        # 2. OOD scoring
        ood_result = self.ood_detector.score(embedding)

        # 3. Cross-validation (if DeQA predictions available)
        if self.cross_validator.has_prediction(image_id, dimension):
            cross_val = self.cross_validator.validate(
                image_id=image_id,
                dimension=dimension,
                siglip2_mu=siglip2_mu,
                siglip2_sigma_sq=siglip2_sigma_sq,
            )
            # 4. Fusion decision (with optional spread)
            decision = self.fusion.decide(
                cross_val, ood_result.mahalanobis_distance, spread_result
            )
        else:
            # No DeQA prediction — use OOD score only
            from .cross_validator import CrossValidationResult
            from .discrete_metrics import discrete_entropy

            dummy_cross_val = CrossValidationResult(
                image_id=image_id,
                dimension=dimension,
                siglip2_probs=level_probs,
                siglip2_mu=float(np.clip(siglip2_mu, 1.0, 5.0)),
                siglip2_sigma_sq=siglip2_sigma_sq,
                deqa_probs=level_probs,  # self-agreement = JSD 0
                deqa_mos=level_probs_to_mos(level_probs),
                cross_model_jsd=0.0,
                mos_delta=0.0,
                siglip2_entropy=discrete_entropy(level_probs),
            )
            decision = self.fusion.decide(
                dummy_cross_val, ood_result.mahalanobis_distance, spread_result
            )

        mos = level_probs_to_mos(level_probs)
        std = level_probs_to_std(level_probs)

        return PseudoLabelSample(
            image_id=image_id,
            dimension=dimension,
            level_probs=level_probs,
            mos=mos,
            std=std,
            confidence_weight=decision.confidence_weight,
            tier=decision.tier,
            decision=decision,
        )

    def process_batch(
        self,
        siglip2_outputs: list[dict],
        embeddings: np.ndarray,
        dimensions: tuple[str, ...] = DIMENSIONS,
        model_predictions: list[dict[str, float]] | None = None,
    ) -> list[PseudoLabelSample]:
        """Process a batch of images through the full pipeline.

        Handles VLM tier-2 validation in a second pass for efficiency.

        Args:
            siglip2_outputs: List of dicts with keys per dimension:
                {
                    "image_id": str,
                    "overall_mu": float, "overall_sigma_sq": float,
                    "sharpness_mu": float, "sharpness_sigma_sq": float,
                    "color_mu": float, "color_sigma_sq": float,
                }
            embeddings: Shape (N, 768) array of SigLIP2 embeddings.
            dimensions: Which dimensions to process.
            model_predictions: Optional list of {model_name: raw_score} per
                image, aligned with siglip2_outputs. Used for spread computation
                when spread_computer is configured.

        Returns:
            List of PseudoLabelSample for all image+dimension combinations.
        """
        all_samples: list[PseudoLabelSample] = []

        # Pre-compute spread per image (shared across dimensions)
        spread_results: list[SpreadResult | None] = [None] * len(siglip2_outputs)
        if self.spread_computer is not None and model_predictions is not None:
            if len(model_predictions) != len(siglip2_outputs):
                raise ValueError(
                    f"model_predictions length ({len(model_predictions)}) must match "
                    f"siglip2_outputs length ({len(siglip2_outputs)})"
                )
            for i, preds in enumerate(model_predictions):
                try:
                    spread_results[i] = self.spread_computer.compute(preds)
                except ValueError:
                    logger.debug(
                        "Spread skipped for image %d: insufficient models", i
                    )

        # Pass 1: Process all without VLM
        for i, output in enumerate(siglip2_outputs):
            image_id = output["image_id"]
            embedding = embeddings[i]

            for dim in dimensions:
                mu = output.get(f"{dim}_mu")
                sigma_sq = output.get(f"{dim}_sigma_sq")
                if mu is None or sigma_sq is None:
                    continue

                sample = self.process_single(
                    image_id=image_id,
                    siglip2_mu=mu,
                    siglip2_sigma_sq=sigma_sq,
                    embedding=embedding,
                    dimension=dim,
                    spread_result=spread_results[i],
                )
                all_samples.append(sample)

        # Pass 2: VLM veto for tier-2 candidates
        if self.vlm_validator is not None:
            tier2_candidates = [
                s for s in all_samples if s.tier == AcceptanceTier.TIER2_TRIGGER
            ]
            if tier2_candidates:
                self._apply_vlm_veto(tier2_candidates, len(siglip2_outputs))

        # Summary stats
        tier_counts = {}
        for s in all_samples:
            tier_counts[s.tier.value] = tier_counts.get(s.tier.value, 0) + 1
        logger.info(
            "Pseudo-label batch: %d samples, tiers: %s", len(all_samples), tier_counts
        )

        return all_samples

    def _apply_vlm_veto(
        self,
        candidates: list[PseudoLabelSample],
        total_pool_size: int,
    ) -> None:
        """Apply VLM veto to tier-2 candidates in-place.

        Args:
            candidates: Tier-2 samples to validate.
            total_pool_size: Total number of images for budget cap.
        """
        if self.vlm_validator is None:
            return

        # Select which to actually send (respecting 10% cap)
        queue_input = [
            {
                "image_id": s.image_id,
                "jsd": s.decision.signals.cross_model_jsd,
                "idx": i,
            }
            for i, s in enumerate(candidates)
        ]
        selected = self.vlm_validator.select_tier2_queue(queue_input, total_pool_size)
        selected_indices = {item["idx"] for item in selected}

        import os

        for i, sample in enumerate(candidates):
            if i not in selected_indices:
                continue

            image_path = os.path.join(self.image_root, sample.image_id)
            if not os.path.exists(image_path):
                logger.warning("Image not found for VLM: %s", image_path)
                continue

            result = self.vlm_validator.validate_single(
                image_id=sample.image_id,
                image_path=image_path,
                dimension=sample.dimension,
                siglip2_mu=sample.mos,
            )

            if result.is_vetoed:
                sample.vlm_vetoed = True
                sample.confidence_weight = 0.0
                logger.debug(
                    "VLM vetoed %s/%s: VLM=%s(%.1f) vs SigLIP2=%.2f",
                    sample.image_id,
                    sample.dimension,
                    result.vlm_label,
                    result.vlm_score or 0,
                    sample.mos,
                )

    def filter_accepted(
        self,
        samples: list[PseudoLabelSample],
        min_weight: float = 0.3,
    ) -> list[PseudoLabelSample]:
        """Filter samples to those suitable for training.

        Args:
            samples: All processed samples.
            min_weight: Minimum confidence weight to include.

        Returns:
            Filtered list excluding hard rejects, VLM vetoes, and low-weight.
        """
        accepted = [
            s
            for s in samples
            if s.confidence_weight >= min_weight
            and s.tier != AcceptanceTier.HARD_REJECT
            and not s.vlm_vetoed
        ]
        logger.info(
            "Filtered: %d/%d samples accepted (min_weight=%.2f)",
            len(accepted),
            len(samples),
            min_weight,
        )
        return accepted
