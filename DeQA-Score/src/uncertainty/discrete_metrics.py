"""Discrete distribution metrics for 5-level quality distributions.

All functions expect level_probs in DeQA convention:
    [excellent, good, fair, poor, bad] = indices [0, 1, 2, 3, 4]

Provides JSD, KL divergence, Shannon entropy, and BALD score for
cross-model comparison and active learning sample selection.
"""

from __future__ import annotations

import numpy as np

# Small constant for numerical stability in log operations
_EPS = 1e-12


def _ensure_valid_dist(p: np.ndarray) -> np.ndarray:
    """Ensure p is a valid probability distribution."""
    p = np.asarray(p, dtype=np.float64).ravel()
    if p.shape[0] != 5:
        msg = f"Expected 5-element distribution, got shape {p.shape}"
        raise ValueError(msg)
    if np.any(p < -_EPS):
        msg = f"Negative probabilities found: {p}"
        raise ValueError(msg)
    p = np.maximum(p, 0.0)
    total = p.sum()
    if total < _EPS:
        msg = "All-zero distribution"
        raise ValueError(msg)
    return p / total


def discrete_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence KL(p || q) for 5-element discrete distributions.

    Args:
        p: Reference distribution (5,).
        q: Approximating distribution (5,).

    Returns:
        KL(p || q) in nats. Returns inf if q has zero mass where p is nonzero.
    """
    p = _ensure_valid_dist(p)
    q = _ensure_valid_dist(q)

    # Only compute where p > 0
    mask = p > _EPS
    if not np.all(q[mask] > _EPS):
        return float("inf")

    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def discrete_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two 5-element distributions.

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m), where m = (p+q)/2.

    Args:
        p: First distribution (5,).
        q: Second distribution (5,).

    Returns:
        JSD in nats. Bounded in [0, ln(2) ≈ 0.693].
    """
    p = _ensure_valid_dist(p)
    q = _ensure_valid_dist(q)
    m = 0.5 * (p + q)

    # Both p and q have mass wherever m does, so KL is always finite
    kl_pm = float(np.sum(p * np.log((p + _EPS) / (m + _EPS))))
    kl_qm = float(np.sum(q * np.log((q + _EPS) / (m + _EPS))))

    return 0.5 * kl_pm + 0.5 * kl_qm


def discrete_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) for a 5-element distribution.

    Args:
        p: Distribution (5,).

    Returns:
        Entropy in nats. Max = ln(5) ≈ 1.609 for uniform distribution.
    """
    p = _ensure_valid_dist(p)
    mask = p > _EPS
    return float(-np.sum(p[mask] * np.log(p[mask])))


def bald_score(distributions: list[np.ndarray]) -> float:
    """BALD (Bayesian Active Learning by Disagreement) score.

    Measures epistemic uncertainty as the gap between the entropy of the
    mean prediction and the mean entropy of individual predictions.

    BALD = H[avg(distributions)] - avg[H(each distribution)]

    High BALD means models agree on being uncertain about different things
    (epistemic uncertainty), as opposed to all being uncertain about the
    same thing (aleatoric uncertainty).

    Args:
        distributions: List of K distributions, each shape (5,).
            Typically [siglip2_probs, deqa_probs] for 2-model case.

    Returns:
        BALD score in nats. Non-negative.
    """
    if len(distributions) < 2:
        return 0.0

    validated = [_ensure_valid_dist(d) for d in distributions]
    avg_dist = np.mean(validated, axis=0)

    entropy_of_avg = discrete_entropy(avg_dist)
    avg_of_entropies = float(np.mean([discrete_entropy(d) for d in validated]))

    # BALD is non-negative by Jensen's inequality
    return max(0.0, entropy_of_avg - avg_of_entropies)
