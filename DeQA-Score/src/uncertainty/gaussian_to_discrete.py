"""Convert SigLIP2's continuous (μ, σ²) output to discrete level_probs.

Uses Gaussian CDF integration over quality-level bins, matching the soft-label
construction in gen_soft_label.py. Also provides the binary (sparse) soft-label
method used when σ is very small.

Level ordering follows the DeQA convention throughout:
    [excellent, good, fair, poor, bad] = indices [0, 1, 2, 3, 4]
    with corresponding scores [5, 4, 3, 2, 1].

This matches:
    - gen_soft_label.py line 76: probs[::-1] "should start with excellent"
    - gen_soft_label.py line 122: range(5, 0, -1)
    - cal_score: np.inner(probs, [5, 4, 3, 2, 1])
    - loss.py line 25: level_names=["excellent", "good", "fair", "poor", "bad"]
"""

from __future__ import annotations

import math

import numpy as np

# DeQA convention: index 0 = excellent (5), index 4 = bad (1)
LEVELS = np.array([5, 4, 3, 2, 1], dtype=np.float64)
LEVEL_WEIGHTS = LEVELS  # alias for clarity in inner products


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF without scipy dependency."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def gaussian_to_level_probs(mu: float, sigma: float) -> np.ndarray:
    """Convert continuous (μ, σ) to discrete 5-level probability distribution.

    Integrates a Gaussian N(μ, σ²) over bins centered at [5, 4, 3, 2, 1].
    Bin boundaries: [level-0.5, level+0.5], clipped to [0.5, 5.5].
    Output ordering: [excellent, good, fair, poor, bad].

    μ is clamped to [1.0, 5.0] before discretization since SigLIP2's
    regression heads are unconstrained and may produce out-of-range values.

    Args:
        mu: Predicted quality score (will be clamped to [1.0, 5.0]).
        sigma: Standard deviation. Floor at 0.1 to avoid degenerate distributions.

    Returns:
        Array of shape (5,) with probabilities summing to 1.0.
        Order: [excellent, good, fair, poor, bad].
    """
    mu = float(np.clip(mu, 1.0, 5.0))
    sigma = max(float(sigma), 0.1)

    probs = np.empty(5, dtype=np.float64)
    for i, level in enumerate(LEVELS):
        lo = max(0.5, level - 0.5)
        hi = min(5.5, level + 0.5)
        probs[i] = _norm_cdf(hi, mu, sigma) - _norm_cdf(lo, mu, sigma)

    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        # Degenerate case: put all mass on nearest level
        nearest = int(np.argmin(np.abs(LEVELS - mu)))
        probs[:] = 0.0
        probs[nearest] = 1.0

    return probs


def binary_level_probs(mu: float) -> np.ndarray:
    """Sparse (binary) soft label using linear interpolation between two nearest levels.

    Matches get_binary_probs() in gen_soft_label.py. Used when σ is very small
    and Gaussian CDF produces near-degenerate distributions.

    Args:
        mu: Quality score, will be clamped to [1.0, 5.0].

    Returns:
        Array of shape (5,) with exactly 2 non-zero entries (or 1 if mu is
        exactly at a level center). Order: [excellent, good, fair, poor, bad].
    """
    mu = float(np.clip(mu, 1.0, 5.0))

    # Bin boundaries at [1.0, 2.0, 3.0, 4.0, 5.0]
    # Find which two levels mu falls between
    probs = np.zeros(5, dtype=np.float64)

    # Map mu to level indices: level 5=idx0, 4=idx1, 3=idx2, 2=idx3, 1=idx4
    # Equivalent to gen_soft_label.py get_binary_probs with reversed output
    for idx in range(4):
        # Boundaries in score space (descending): 5→4→3→2→1
        upper = 5.0 - idx  # 5, 4, 3, 2
        lower = 4.0 - idx  # 4, 3, 2, 1
        if lower <= mu <= upper:
            frac_upper = (mu - lower) / (upper - lower)
            probs[idx] = frac_upper
            probs[idx + 1] = 1.0 - frac_upper
            return probs

    # Edge case: mu exactly at 1.0 or 5.0
    if mu >= 4.5:
        probs[0] = 1.0
    else:
        probs[4] = 1.0
    return probs


def siglip2_output_to_level_probs(
    mu: float,
    sigma_sq: float,
    sigma_floor: float = 0.1,
    binary_sigma_threshold: float = 0.15,
) -> np.ndarray:
    """Convert SigLIP2's (μ, σ²) output directly to level_probs.

    Uses Gaussian CDF method by default, falls back to binary interpolation
    when σ is very small (matching gen_soft_label.py's thre_std behavior).

    Args:
        mu: Predicted quality score from SigLIP2.
        sigma_sq: Predicted variance from GaussianNLL head.
        sigma_floor: Minimum σ value to prevent degenerate distributions.
        binary_sigma_threshold: Below this σ, use binary interpolation instead.

    Returns:
        Array of shape (5,) in DeQA convention [excellent→bad].
    """
    sigma = max(math.sqrt(max(sigma_sq, 0.0)), sigma_floor)
    if sigma < binary_sigma_threshold:
        return binary_level_probs(mu)
    return gaussian_to_level_probs(mu, sigma)


def level_probs_to_mos(probs: np.ndarray) -> float:
    """Reconstruct MOS from level_probs using DeQA's weighted sum.

    Equivalent to cal_score's np.inner(probs, [5, 4, 3, 2, 1]).

    Args:
        probs: Shape (5,) array in DeQA convention [excellent→bad].

    Returns:
        Scalar MOS score in [1.0, 5.0].
    """
    return float(np.inner(np.asarray(probs), LEVEL_WEIGHTS))


def level_probs_to_std(probs: np.ndarray) -> float:
    """Compute standard deviation from level_probs distribution.

    Equivalent to cal_std in cal_distribution_gap.py.

    Args:
        probs: Shape (5,) array in DeQA convention [excellent→bad].

    Returns:
        Standard deviation of the discrete distribution.
    """
    probs = np.asarray(probs)
    mos = level_probs_to_mos(probs)
    variance = float(np.inner(probs, (LEVEL_WEIGHTS - mos) ** 2))
    return math.sqrt(variance)
