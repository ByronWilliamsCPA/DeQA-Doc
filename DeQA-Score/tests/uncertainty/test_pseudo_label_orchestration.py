"""Tests for pseudo_label pipeline orchestration.

Tests process_single, process_batch, filter_accepted, and VLM veto wiring
with fully mocked components (no GPU or API calls needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.uncertainty.fusion import AcceptanceDecision, AcceptanceTier, UncertaintySignals
from src.uncertainty.pseudo_label import PseudoLabelPipeline, PseudoLabelSample


def _make_mock_pipeline(
    ood_distance: float = 10.0,
    has_deqa: bool = True,
    tier: AcceptanceTier = AcceptanceTier.AUTO_ACCEPT,
    weight: float = 0.9,
) -> PseudoLabelPipeline:
    """Create pipeline with mocked dependencies."""
    # Mock OOD detector
    ood_detector = MagicMock()
    ood_result = MagicMock()
    ood_result.mahalanobis_distance = ood_distance
    ood_detector.score.return_value = ood_result

    # Mock cross validator
    cross_validator = MagicMock()
    cross_validator.has_prediction.return_value = has_deqa
    cross_val_result = MagicMock()
    cross_validator.validate.return_value = cross_val_result

    # Mock fusion
    fusion = MagicMock()
    signals = UncertaintySignals(
        mahalanobis_distance=ood_distance,
        cross_model_jsd=0.05,
        siglip2_sigma_sq=0.1,
        siglip2_entropy=0.5,
    )
    decision = AcceptanceDecision(
        image_id="mock",
        dimension="overall",
        tier=tier,
        confidence_weight=weight,
        signals=signals,
        reason="test",
    )
    fusion.decide.return_value = decision

    return PseudoLabelPipeline(
        ood_detector=ood_detector,
        cross_validator=cross_validator,
        fusion=fusion,
    )


class TestProcessSingle:
    """Tests for process_single orchestration."""

    def test_returns_pseudo_label_sample(self) -> None:
        pipeline = _make_mock_pipeline()
        embedding = np.random.randn(768).astype(np.float32)
        sample = pipeline.process_single(
            image_id="test.jpg",
            siglip2_mu=3.5,
            siglip2_sigma_sq=0.2,
            embedding=embedding,
            dimension="overall",
        )
        assert isinstance(sample, PseudoLabelSample)
        assert sample.image_id == "test.jpg"
        assert sample.dimension == "overall"
        assert len(sample.level_probs) == 5
        assert 1.0 <= sample.mos <= 5.0

    def test_calls_ood_detector(self) -> None:
        pipeline = _make_mock_pipeline()
        embedding = np.random.randn(768).astype(np.float32)
        pipeline.process_single("img.jpg", 3.0, 0.1, embedding, "overall")
        pipeline.ood_detector.score.assert_called_once_with(embedding)

    def test_calls_cross_validator_when_available(self) -> None:
        pipeline = _make_mock_pipeline(has_deqa=True)
        embedding = np.random.randn(768).astype(np.float32)
        pipeline.process_single("img.jpg", 3.0, 0.1, embedding, "overall")
        pipeline.cross_validator.validate.assert_called_once()

    def test_skips_cross_validator_when_unavailable(self) -> None:
        pipeline = _make_mock_pipeline(has_deqa=False)
        embedding = np.random.randn(768).astype(np.float32)
        pipeline.process_single("img.jpg", 3.0, 0.1, embedding, "overall")
        pipeline.cross_validator.validate.assert_not_called()
        # Fusion still called (with dummy cross-val result)
        pipeline.fusion.decide.assert_called_once()

    def test_tier_propagated(self) -> None:
        pipeline = _make_mock_pipeline(tier=AcceptanceTier.HARD_REJECT, weight=0.0)
        embedding = np.random.randn(768).astype(np.float32)
        sample = pipeline.process_single("img.jpg", 3.0, 0.1, embedding, "overall")
        assert sample.tier == AcceptanceTier.HARD_REJECT
        assert sample.confidence_weight == 0.0


class TestProcessBatch:
    """Tests for process_batch orchestration."""

    def test_processes_all_dimensions(self) -> None:
        pipeline = _make_mock_pipeline()
        outputs = [
            {
                "image_id": "img1.jpg",
                "overall_mu": 3.5,
                "overall_sigma_sq": 0.2,
                "sharpness_mu": 4.0,
                "sharpness_sigma_sq": 0.1,
                "color_mu": 3.0,
                "color_sigma_sq": 0.3,
            }
        ]
        embeddings = np.random.randn(1, 768).astype(np.float32)
        samples = pipeline.process_batch(outputs, embeddings)
        assert len(samples) == 3  # 3 dimensions
        dims = {s.dimension for s in samples}
        assert dims == {"overall", "sharpness", "color"}

    def test_skips_missing_dimensions(self) -> None:
        pipeline = _make_mock_pipeline()
        outputs = [
            {
                "image_id": "img1.jpg",
                "overall_mu": 3.5,
                "overall_sigma_sq": 0.2,
                # sharpness and color missing
            }
        ]
        embeddings = np.random.randn(1, 768).astype(np.float32)
        samples = pipeline.process_batch(outputs, embeddings)
        assert len(samples) == 1
        assert samples[0].dimension == "overall"

    def test_multiple_images(self) -> None:
        pipeline = _make_mock_pipeline()
        outputs = [
            {"image_id": f"img{i}.jpg", "overall_mu": 3.0, "overall_sigma_sq": 0.2}
            for i in range(5)
        ]
        embeddings = np.random.randn(5, 768).astype(np.float32)
        samples = pipeline.process_batch(outputs, embeddings, dimensions=("overall",))
        assert len(samples) == 5


class TestFilterAccepted:
    """Tests for filter_accepted."""

    def _make_sample(
        self,
        weight: float = 0.9,
        tier: AcceptanceTier = AcceptanceTier.AUTO_ACCEPT,
        vetoed: bool = False,
    ) -> PseudoLabelSample:
        signals = UncertaintySignals(
            mahalanobis_distance=10.0,
            cross_model_jsd=0.05,
            siglip2_sigma_sq=0.1,
            siglip2_entropy=0.5,
        )
        return PseudoLabelSample(
            image_id="test.jpg",
            dimension="overall",
            level_probs=np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            mos=3.0,
            std=0.5,
            confidence_weight=weight,
            tier=tier,
            decision=AcceptanceDecision(image_id="test.jpg", dimension="overall", tier=tier, confidence_weight=weight, signals=signals, reason="test"),
            vlm_vetoed=vetoed,
        )

    def test_accepts_high_weight(self) -> None:
        pipeline = _make_mock_pipeline()
        samples = [self._make_sample(weight=0.9)]
        accepted = pipeline.filter_accepted(samples, min_weight=0.3)
        assert len(accepted) == 1

    def test_rejects_low_weight(self) -> None:
        pipeline = _make_mock_pipeline()
        samples = [self._make_sample(weight=0.1)]
        accepted = pipeline.filter_accepted(samples, min_weight=0.3)
        assert len(accepted) == 0

    def test_rejects_hard_reject_tier(self) -> None:
        pipeline = _make_mock_pipeline()
        samples = [self._make_sample(tier=AcceptanceTier.HARD_REJECT, weight=0.9)]
        accepted = pipeline.filter_accepted(samples, min_weight=0.3)
        assert len(accepted) == 0

    def test_rejects_vlm_vetoed(self) -> None:
        pipeline = _make_mock_pipeline()
        samples = [self._make_sample(vetoed=True)]
        accepted = pipeline.filter_accepted(samples, min_weight=0.3)
        assert len(accepted) == 0

    def test_mixed_filtering(self) -> None:
        pipeline = _make_mock_pipeline()
        samples = [
            self._make_sample(weight=0.9),  # accepted
            self._make_sample(weight=0.1),  # rejected: low weight
            self._make_sample(tier=AcceptanceTier.HARD_REJECT, weight=0.9),  # rejected
            self._make_sample(vetoed=True),  # rejected: vetoed
        ]
        accepted = pipeline.filter_accepted(samples, min_weight=0.3)
        assert len(accepted) == 1
