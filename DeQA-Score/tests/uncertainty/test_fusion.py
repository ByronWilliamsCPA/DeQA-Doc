"""Tests for fusion.py."""

import numpy as np
import pytest

from src.uncertainty.cross_validator import CrossValidationResult
from src.uncertainty.fusion import (
    AcceptanceTier,
    JSDThresholds,
    UncertaintyFusion,
)


@pytest.fixture
def fusion():
    return UncertaintyFusion()


def _make_cross_val(
    jsd: float = 0.03,
    sigma_sq: float = 0.3,
    entropy: float = 0.8,
    image_id: str = "test.jpg",
    dimension: str = "overall",
) -> CrossValidationResult:
    """Helper to create CrossValidationResult with specified signals."""
    return CrossValidationResult(
        image_id=image_id,
        dimension=dimension,
        siglip2_probs=np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
        siglip2_mu=3.2,
        siglip2_sigma_sq=sigma_sq,
        deqa_probs=np.array([0.1, 0.3, 0.4, 0.15, 0.05]),
        deqa_mos=3.2,
        cross_model_jsd=jsd,
        mos_delta=0.0,
        siglip2_entropy=entropy,
    )


class TestAcceptanceTiers:
    def test_hard_reject(self, fusion):
        """d_M > 58.6 → hard reject."""
        cv = _make_cross_val(jsd=0.01, sigma_sq=0.1, entropy=0.5)
        decision = fusion.decide(cv, mahalanobis_distance=60.0)
        assert decision.tier == AcceptanceTier.HARD_REJECT
        assert decision.confidence_weight == 0.0

    def test_ood_trigger(self, fusion):
        """46.0 < d_M <= 58.6 → tier-2 trigger."""
        cv = _make_cross_val(jsd=0.01, sigma_sq=0.1, entropy=0.5)
        decision = fusion.decide(cv, mahalanobis_distance=50.0)
        assert decision.tier == AcceptanceTier.TIER2_TRIGGER

    def test_auto_accept(self, fusion):
        """All signals below auto thresholds → auto accept."""
        cv = _make_cross_val(jsd=0.03, sigma_sq=0.3, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.AUTO_ACCEPT
        assert decision.confidence_weight == 1.0

    def test_low_weight(self, fusion):
        """Signals between auto and low thresholds → low weight."""
        cv = _make_cross_val(jsd=0.09, sigma_sq=0.8, entropy=1.3)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.LOW_WEIGHT
        assert 0.3 <= decision.confidence_weight <= 0.6

    def test_tier2_on_high_jsd(self, fusion):
        """JSD > low threshold → tier-2 trigger."""
        cv = _make_cross_val(jsd=0.15, sigma_sq=0.3, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.TIER2_TRIGGER

    def test_tier2_on_high_sigma(self, fusion):
        """σ² > low threshold → tier-2 trigger."""
        cv = _make_cross_val(jsd=0.03, sigma_sq=1.5, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.TIER2_TRIGGER

    def test_tier2_on_high_entropy(self, fusion):
        """Entropy > low threshold → tier-2 trigger."""
        cv = _make_cross_val(jsd=0.03, sigma_sq=0.3, entropy=1.6)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.TIER2_TRIGGER


class TestPerDimensionJSD:
    def test_custom_thresholds(self):
        """Per-dimension JSD thresholds should be respected."""
        fusion = UncertaintyFusion(
            jsd_thresholds={
                "overall": JSDThresholds(auto_accept=0.05, low_weight=0.10),
                "sharpness": JSDThresholds(auto_accept=0.10, low_weight=0.20),
                "color": JSDThresholds(auto_accept=0.03, low_weight=0.06),
            }
        )

        # JSD=0.07: auto for sharpness, low for overall, tier2 for color
        cv_overall = _make_cross_val(jsd=0.07, dimension="overall")
        cv_sharpness = _make_cross_val(jsd=0.07, dimension="sharpness")
        cv_color = _make_cross_val(jsd=0.07, dimension="color")

        d_overall = fusion.decide(cv_overall, 30.0)
        d_sharpness = fusion.decide(cv_sharpness, 30.0)
        d_color = fusion.decide(cv_color, 30.0)

        assert d_overall.tier == AcceptanceTier.LOW_WEIGHT
        assert d_sharpness.tier == AcceptanceTier.AUTO_ACCEPT
        assert d_color.tier == AcceptanceTier.TIER2_TRIGGER


class TestDecisionMetadata:
    def test_decision_has_signals(self, fusion):
        cv = _make_cross_val(jsd=0.03, sigma_sq=0.3, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.signals.mahalanobis_distance == 30.0
        assert decision.signals.cross_model_jsd == 0.03
        assert decision.signals.siglip2_sigma_sq == 0.3
        assert decision.signals.siglip2_entropy == 0.8
        assert decision.image_id == "test.jpg"
        assert decision.dimension == "overall"

    def test_decision_has_reason(self, fusion):
        cv = _make_cross_val()
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0


class TestSigmaSqThreshold:
    def test_sigma_sq_064_is_auto_accept(self, fusion):
        """σ²=0.64 (= 0.8²) should pass auto-accept."""
        cv = _make_cross_val(jsd=0.03, sigma_sq=0.64, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.AUTO_ACCEPT

    def test_sigma_sq_065_is_low_weight(self, fusion):
        """σ²=0.65 slightly above 0.64 → low weight."""
        cv = _make_cross_val(jsd=0.03, sigma_sq=0.65, entropy=0.8)
        decision = fusion.decide(cv, mahalanobis_distance=30.0)
        assert decision.tier == AcceptanceTier.LOW_WEIGHT
