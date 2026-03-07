"""Tests for gaussian_to_discrete.py."""

import math

import numpy as np

from src.uncertainty.gaussian_to_discrete import (
    LEVELS,
    LEVEL_WEIGHTS,
    binary_level_probs,
    gaussian_to_level_probs,
    level_probs_to_mos,
    level_probs_to_std,
    siglip2_output_to_level_probs,
)


class TestLevelOrdering:
    """Verify DeQA convention: [excellent(5), good(4), fair(3), poor(2), bad(1)]."""

    def test_levels_array(self):
        np.testing.assert_array_equal(LEVELS, [5, 4, 3, 2, 1])

    def test_level_weights_match_levels(self):
        np.testing.assert_array_equal(LEVEL_WEIGHTS, LEVELS)

    def test_excellent_is_index_0(self):
        """For high mu (4.7), most mass should be at index 0 (excellent)."""
        probs = gaussian_to_level_probs(4.7, 0.3)
        assert np.argmax(probs) == 0, (
            f"Expected argmax=0 (excellent), got {np.argmax(probs)}"
        )

    def test_bad_is_index_4(self):
        """For low mu (1.5), most mass should be at index 4 (bad)."""
        probs = gaussian_to_level_probs(1.5, 0.3)
        assert np.argmax(probs) == 4, f"Expected argmax=4 (bad), got {np.argmax(probs)}"

    def test_fair_is_index_2(self):
        """For mu=3.0, most mass should be at index 2 (fair)."""
        probs = gaussian_to_level_probs(3.0, 0.3)
        assert np.argmax(probs) == 2, (
            f"Expected argmax=2 (fair), got {np.argmax(probs)}"
        )


class TestGaussianToLevelProbs:
    """Test Gaussian CDF discretization."""

    def test_sums_to_one(self):
        for mu in [1.0, 2.5, 3.7, 5.0]:
            for sigma in [0.3, 0.5, 0.8, 1.2]:
                probs = gaussian_to_level_probs(mu, sigma)
                assert abs(probs.sum() - 1.0) < 1e-10, (
                    f"mu={mu}, sigma={sigma}: sum={probs.sum()}"
                )

    def test_all_nonnegative(self):
        for mu in [0.5, 1.0, 3.0, 5.0, 5.5]:
            probs = gaussian_to_level_probs(mu, 0.8)
            assert np.all(probs >= 0), f"Negative probs for mu={mu}: {probs}"

    def test_shape(self):
        probs = gaussian_to_level_probs(3.0, 0.8)
        assert probs.shape == (5,)

    def test_mu_clamping_low(self):
        """mu=0.5 should be clamped to 1.0."""
        probs_clamped = gaussian_to_level_probs(0.5, 0.8)
        probs_at_1 = gaussian_to_level_probs(1.0, 0.8)
        np.testing.assert_array_almost_equal(probs_clamped, probs_at_1)

    def test_mu_clamping_high(self):
        """mu=5.8 should be clamped to 5.0."""
        probs_clamped = gaussian_to_level_probs(5.8, 0.8)
        probs_at_5 = gaussian_to_level_probs(5.0, 0.8)
        np.testing.assert_array_almost_equal(probs_clamped, probs_at_5)

    def test_sigma_floor(self):
        """sigma=0.01 should be floored to 0.1."""
        probs = gaussian_to_level_probs(3.0, 0.01)
        assert probs.sum() > 0.999

    def test_reference_value(self):
        """Verify against plan reference: mu=3.7, sigma=0.8.

        index 0=excellent should be ≈0.235, index 4=bad should be ≈0.001.
        """
        probs = gaussian_to_level_probs(3.7, 0.8)
        assert probs[0] > 0.14, f"Excellent prob too low: {probs[0]}"
        assert probs[4] < 0.01, f"Bad prob too high: {probs[4]}"
        # Most mass should be on good (index 1)
        assert np.argmax(probs) == 1, (
            f"Expected argmax=1 (good), got {np.argmax(probs)}"
        )

    def test_mos_reconstruction(self):
        """np.inner(probs, [5,4,3,2,1]) should approximate the original mu."""
        for mu in [1.5, 2.0, 3.0, 3.7, 4.5]:
            probs = gaussian_to_level_probs(mu, 0.8)
            reconstructed = level_probs_to_mos(probs)
            # Allow tolerance since discretization loses information
            assert abs(reconstructed - mu) < 0.5, (
                f"mu={mu}, reconstructed={reconstructed:.3f}"
            )


class TestBinaryLevelProbs:
    """Test sparse binary soft labels."""

    def test_sums_to_one(self):
        for mu in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            probs = binary_level_probs(mu)
            assert abs(probs.sum() - 1.0) < 1e-10

    def test_at_most_two_nonzero(self):
        for mu in [1.3, 2.7, 3.5, 4.1]:
            probs = binary_level_probs(mu)
            assert np.count_nonzero(probs) <= 2

    def test_exact_level(self):
        """At exact level centers, only one bin should have mass."""
        # mu=5.0 → all mass on excellent (index 0)
        probs = binary_level_probs(5.0)
        assert probs[0] == 1.0

    def test_midpoint(self):
        """mu=3.5 → equal mass on good (index 1) and fair (index 2)."""
        probs = binary_level_probs(3.5)
        assert abs(probs[1] - 0.5) < 1e-10
        assert abs(probs[2] - 0.5) < 1e-10


class TestSiglip2OutputToLevelProbs:
    """Test the convenience wrapper for SigLIP2 outputs."""

    def test_basic(self):
        probs = siglip2_output_to_level_probs(3.5, 0.64)
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_negative_sigma_sq(self):
        """Negative σ² should be handled gracefully (floored to 0)."""
        probs = siglip2_output_to_level_probs(3.0, -0.1)
        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_very_small_sigma_uses_binary(self):
        """Very small σ² should produce binary (sparse) labels."""
        probs = siglip2_output_to_level_probs(3.5, 0.01)
        assert np.count_nonzero(probs) <= 2


class TestLevelProbsToMos:
    """Test MOS reconstruction."""

    def test_uniform(self):
        """Uniform probs → MOS = 3.0."""
        mos = level_probs_to_mos(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        assert abs(mos - 3.0) < 1e-10

    def test_all_excellent(self):
        """All mass on excellent → MOS = 5.0."""
        mos = level_probs_to_mos(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        assert abs(mos - 5.0) < 1e-10

    def test_all_bad(self):
        """All mass on bad → MOS = 1.0."""
        mos = level_probs_to_mos(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
        assert abs(mos - 1.0) < 1e-10


class TestLevelProbsToStd:
    """Test standard deviation computation."""

    def test_degenerate(self):
        """All mass on one level → std = 0."""
        std = level_probs_to_std(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        assert abs(std) < 1e-10

    def test_uniform(self):
        """Uniform distribution → std = sqrt(2) ≈ 1.414."""
        std = level_probs_to_std(np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        expected = math.sqrt(2.0)  # sqrt(sum(0.2 * (level-3)^2) for levels 5,4,3,2,1)
        assert abs(std - expected) < 1e-10
