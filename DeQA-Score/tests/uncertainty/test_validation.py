"""Tests for validation.py."""

import numpy as np
import pytest

from src.uncertainty.validation import (
    BootstrapCI,
    HarmCheckResult,
    PseudoLabelValidator,
    _plcc,
    _srcc,
)


class TestSRCC:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_srcc(x, x) - 1.0) < 1e-10

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(_srcc(x, y) - (-1.0)) < 1e-10

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        y = rng.standard_normal(1000)
        assert abs(_srcc(x, y)) < 0.1

    def test_short_array(self):
        assert _srcc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


class TestPLCC:
    def test_perfect_positive(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_plcc(x, x) - 1.0) < 1e-10

    def test_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(_plcc(x, y) - (-1.0)) < 1e-10

    def test_constant_returns_zero(self):
        x = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _plcc(x, y) == 0.0


class TestBootstrapCI:
    def test_returns_correct_type(self):
        validator = PseudoLabelValidator()
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gt = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = validator.bootstrap_ci(preds, gt, metric_fn="srcc", n_bootstrap=100)
        assert isinstance(result, BootstrapCI)
        assert result.metric_name == "srcc"
        assert result.n_bootstrap == 100

    def test_ci_contains_point_estimate(self):
        validator = PseudoLabelValidator()
        rng = np.random.default_rng(42)
        x = np.arange(100, dtype=np.float64)
        y = x + rng.standard_normal(100) * 5
        result = validator.bootstrap_ci(x, y, n_bootstrap=500)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_plcc_metric(self):
        validator = PseudoLabelValidator()
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gt = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = validator.bootstrap_ci(preds, gt, metric_fn="plcc", n_bootstrap=100)
        assert result.metric_name == "plcc"
        assert result.point_estimate > 0.9

    def test_high_alpha_wider_ci(self):
        validator = PseudoLabelValidator()
        rng = np.random.default_rng(42)
        x = np.arange(50, dtype=np.float64)
        y = x + rng.standard_normal(50) * 5
        ci_95 = validator.bootstrap_ci(x, y, alpha=0.95, n_bootstrap=500)
        ci_80 = validator.bootstrap_ci(x, y, alpha=0.80, n_bootstrap=500)
        width_95 = ci_95.ci_upper - ci_95.ci_lower
        width_80 = ci_80.ci_upper - ci_80.ci_lower
        assert width_95 >= width_80


class TestHarmCheck:
    def test_no_harm(self):
        validator = PseudoLabelValidator(harm_tolerance=0.02)
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        current = np.array([1.05, 2.05, 3.05, 3.95, 5.05])
        result = validator.harm_check(baseline, current, gt)
        assert isinstance(result, HarmCheckResult)
        assert not result.is_harmful

    def test_harm_detected(self):
        validator = PseudoLabelValidator(harm_tolerance=0.02)
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect SRCC=1.0
        current = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Reversed → SRCC=-1.0
        result = validator.harm_check(baseline, current, gt)
        assert result.is_harmful
        assert result.srcc_delta < -0.02

    def test_dimension_stored(self):
        validator = PseudoLabelValidator()
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validator.harm_check(gt, gt, gt, dimension="sharpness")
        assert result.dimension == "sharpness"


class TestSRCCFloorCheck:
    def test_passes(self):
        validator = PseudoLabelValidator(srcc_floor=0.65)
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert validator.check_srcc_floor(preds, gt)

    def test_fails(self):
        validator = PseudoLabelValidator(srcc_floor=0.65)
        rng = np.random.default_rng(42)
        preds = rng.standard_normal(20)
        gt = rng.standard_normal(20)
        assert not validator.check_srcc_floor(preds, gt)


class TestDistributionDrift:
    def test_identical_distributions(self):
        probs = [np.array([0.2, 0.2, 0.2, 0.2, 0.2])] * 10
        drift = PseudoLabelValidator.distribution_drift(probs, probs)
        assert drift["mean_mos_delta"] < 1e-10
        assert drift["mos_ks_statistic"] < 1e-10

    def test_shifted_distributions(self):
        pseudo = [np.array([0.8, 0.15, 0.05, 0.0, 0.0])] * 50
        training = [np.array([0.0, 0.0, 0.05, 0.15, 0.8])] * 50
        drift = PseudoLabelValidator.distribution_drift(pseudo, training)
        assert drift["mean_mos_delta"] > 2.0
        assert drift["mos_ks_statistic"] > 0.9

    def test_per_level_deltas(self):
        pseudo = [np.array([0.5, 0.3, 0.2, 0.0, 0.0])] * 20
        training = [np.array([0.0, 0.0, 0.2, 0.3, 0.5])] * 20
        drift = PseudoLabelValidator.distribution_drift(pseudo, training)
        assert drift["per_level_mean_delta_excellent"] == pytest.approx(0.5, abs=1e-6)
        assert drift["per_level_mean_delta_bad"] == pytest.approx(0.5, abs=1e-6)


class TestPerCategoryAudit:
    def test_flags_high_auto_accept(self):
        samples = [
            {"category": "invoice", "tier": "auto_accept", "confidence_weight": 1.0}
            for _ in range(20)
        ]
        audit = PseudoLabelValidator.per_category_acceptance_audit(samples)
        assert audit["invoice"]["auto_accept_rate"] == 1.0
        assert audit["invoice"]["total"] == 20.0

    def test_mixed_categories(self):
        samples = [
            {"category": "invoice", "tier": "auto_accept", "confidence_weight": 1.0},
            {"category": "invoice", "tier": "low_weight", "confidence_weight": 0.5},
            {"category": "receipt", "tier": "auto_accept", "confidence_weight": 1.0},
        ]
        audit = PseudoLabelValidator.per_category_acceptance_audit(samples)
        assert audit["invoice"]["auto_accept_rate"] == pytest.approx(0.5, abs=1e-4)
        assert audit["receipt"]["auto_accept_rate"] == 1.0
