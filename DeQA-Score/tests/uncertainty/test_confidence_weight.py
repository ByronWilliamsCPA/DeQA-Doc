"""Tests for confidence_weight module."""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.confidence_weight import (
    ENTROPY_RANGE,
    SIGMA_SQ_RANGE,
    confidence_weight_absolute,
    confidence_weights_rank,
)


class TestConfidenceWeightAbsolute:
    """Tests for absolute-mode confidence weighting."""

    def test_low_uncertainty_gives_high_weight(self):
        """Minimum sigma_sq and entropy should produce weight ~1.0."""
        w = confidence_weight_absolute(
            sigma_sq=SIGMA_SQ_RANGE[0],
            entropy=ENTROPY_RANGE[0],
        )
        assert w == pytest.approx(1.0, abs=0.01)

    def test_high_uncertainty_gives_low_weight(self):
        """Maximum sigma_sq and entropy should produce weight ~0.3."""
        w = confidence_weight_absolute(
            sigma_sq=SIGMA_SQ_RANGE[1],
            entropy=ENTROPY_RANGE[1],
        )
        assert w == pytest.approx(0.3, abs=0.01)

    def test_mid_uncertainty_gives_mid_weight(self):
        """Midpoint signals should produce weight ~0.65."""
        mid_sigma = (SIGMA_SQ_RANGE[0] + SIGMA_SQ_RANGE[1]) / 2
        mid_entropy = (ENTROPY_RANGE[0] + ENTROPY_RANGE[1]) / 2
        w = confidence_weight_absolute(sigma_sq=mid_sigma, entropy=mid_entropy)
        assert 0.55 < w < 0.75

    def test_weight_bounded(self):
        """Weight should always be in [min_weight, 1.0]."""
        # Beyond observed range
        w = confidence_weight_absolute(sigma_sq=0.1, entropy=2.0)
        assert 0.3 <= w <= 1.0

        # Negative sigma (edge case)
        w = confidence_weight_absolute(sigma_sq=-0.001, entropy=0.0)
        assert 0.3 <= w <= 1.0

    def test_custom_min_weight(self):
        w = confidence_weight_absolute(
            sigma_sq=SIGMA_SQ_RANGE[1],
            entropy=ENTROPY_RANGE[1],
            min_weight=0.5,
        )
        assert w == pytest.approx(0.5, abs=0.01)

    def test_sigma_only_contributes(self):
        """High sigma_sq alone should reduce weight."""
        w_low = confidence_weight_absolute(sigma_sq=0.002, entropy=0.0)
        w_high = confidence_weight_absolute(sigma_sq=0.019, entropy=0.0)
        assert w_low > w_high

    def test_entropy_only_contributes(self):
        """High entropy alone should reduce weight."""
        w_low = confidence_weight_absolute(sigma_sq=0.002, entropy=0.0)
        w_high = confidence_weight_absolute(sigma_sq=0.002, entropy=0.693)
        assert w_low > w_high


class TestConfidenceWeightsRank:
    """Tests for rank-based confidence weighting."""

    def test_basic_ordering(self):
        """Lower uncertainty samples should get higher weights."""
        sigma_sqs = np.array([0.001, 0.005, 0.010, 0.020])
        entropies = np.array([0.1, 0.3, 0.5, 0.7])
        weights = confidence_weights_rank(sigma_sqs, entropies)

        assert weights[0] > weights[1] > weights[2] > weights[3]

    def test_output_range(self):
        """All weights should be in [min_weight, 1.0]."""
        rng = np.random.default_rng(42)
        sigma_sqs = rng.uniform(0.001, 0.02, size=100)
        entropies = rng.uniform(0.0, 0.7, size=100)
        weights = confidence_weights_rank(sigma_sqs, entropies)

        assert weights.min() >= 0.3 - 1e-6
        assert weights.max() <= 1.0 + 1e-6

    def test_custom_min_weight(self):
        sigma_sqs = np.array([0.001, 0.02])
        entropies = np.array([0.0, 0.7])
        weights = confidence_weights_rank(sigma_sqs, entropies, min_weight=0.5)
        assert weights.min() >= 0.5 - 1e-6

    def test_single_sample(self):
        """Single sample should get weight 1.0."""
        weights = confidence_weights_rank(
            np.array([0.005]), np.array([0.3])
        )
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_identical_signals(self):
        """Identical signals should get identical weights."""
        sigma_sqs = np.array([0.005, 0.005, 0.005])
        entropies = np.array([0.3, 0.3, 0.3])
        weights = confidence_weights_rank(sigma_sqs, entropies)
        assert np.allclose(weights, weights[0])

    def test_monotonic_with_sigma_only(self):
        """Rank weights should be monotonic when only sigma varies."""
        sigma_sqs = np.array([0.002, 0.004, 0.008, 0.016])
        entropies = np.array([0.3, 0.3, 0.3, 0.3])
        weights = confidence_weights_rank(sigma_sqs, entropies)

        diffs = np.diff(weights)
        assert (diffs <= 0).all(), "Weights should decrease with increasing sigma_sq"
