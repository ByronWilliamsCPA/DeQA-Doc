"""Tests for discrete_metrics.py."""

import math

import numpy as np
import pytest

from src.uncertainty.discrete_metrics import (
    bald_score,
    discrete_entropy,
    discrete_jsd,
    discrete_kl,
)


class TestDiscreteKL:
    """Test KL divergence for discrete distributions."""

    def test_identical_distributions(self):
        p = np.array([0.2, 0.3, 0.3, 0.1, 0.1])
        assert discrete_kl(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        q = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        assert discrete_kl(p, q) > 0

    def test_asymmetric(self):
        p = np.array([0.9, 0.1, 0.0, 0.0, 0.0])
        q = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        # KL is asymmetric: KL(p||q) != KL(q||p)
        kl_pq = discrete_kl(p, q)
        kl_qp = discrete_kl(q, p)
        assert kl_pq != pytest.approx(kl_qp, abs=0.01)

    def test_zero_in_q_where_p_positive(self):
        """KL should be inf when q=0 where p>0."""
        p = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        assert discrete_kl(p, q) == float("inf")

    def test_invalid_length(self):
        with pytest.raises(ValueError, match="Expected 5-element"):
            discrete_kl(np.array([0.5, 0.5]), np.array([0.5, 0.5]))


class TestDiscreteJSD:
    """Test Jensen-Shannon divergence."""

    def test_identical_distributions(self):
        p = np.array([0.2, 0.3, 0.3, 0.1, 0.1])
        assert discrete_jsd(p, p) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        q = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        assert discrete_jsd(p, q) == pytest.approx(discrete_jsd(q, p), abs=1e-10)

    def test_maximum_divergence(self):
        """Dirac deltas at opposite ends → JSD = ln(2)."""
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        expected = math.log(2)
        assert discrete_jsd(p, q) == pytest.approx(expected, abs=1e-6)

    def test_bounded(self):
        """JSD should always be in [0, ln(2)]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            jsd = discrete_jsd(p, q)
            assert 0 <= jsd <= math.log(2) + 1e-10


class TestDiscreteEntropy:
    """Test Shannon entropy."""

    def test_uniform(self):
        """Uniform distribution → H = ln(5)."""
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        expected = math.log(5)
        assert discrete_entropy(p) == pytest.approx(expected, abs=1e-10)

    def test_degenerate(self):
        """Dirac delta → H = 0."""
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert discrete_entropy(p) == pytest.approx(0.0, abs=1e-10)

    def test_positive(self):
        p = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        assert discrete_entropy(p) > 0

    def test_bounded(self):
        """Entropy should always be in [0, ln(5)]."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(5))
            h = discrete_entropy(p)
            assert 0 <= h <= math.log(5) + 1e-10


class TestBALD:
    """Test BALD (Bayesian Active Learning by Disagreement) score."""

    def test_identical_distributions(self):
        """If all models agree, BALD = 0."""
        p = np.array([0.2, 0.3, 0.3, 0.1, 0.1])
        assert bald_score([p, p, p]) == pytest.approx(0.0, abs=1e-10)

    def test_maximum_disagreement(self):
        """If models are certain but disagree, BALD is high."""
        p1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        score = bald_score([p1, p2])
        assert score > 0.5  # Should be substantial

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        for _ in range(100):
            dists = [rng.dirichlet(np.ones(5)) for _ in range(3)]
            assert bald_score(dists) >= -1e-10

    def test_single_distribution(self):
        """Single distribution → BALD = 0."""
        p = np.array([0.2, 0.3, 0.3, 0.1, 0.1])
        assert bald_score([p]) == 0.0

    def test_empty(self):
        assert bald_score([]) == 0.0
