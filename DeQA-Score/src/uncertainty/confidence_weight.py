"""Continuous confidence weighting from SigLIP2 uncertainty signals.

Computes a per-sample confidence weight in [min_weight, 1.0] from sigma_sq
and entropy, validated in Step 2.1b: sigma_sq + entropy rank-average reduces
effective MAE by 3.7% while retaining 94% of data.

Two modes:
- Absolute: uses fixed normalization ranges (for online/single-sample use)
- Rank: uses dataset-level percentiles (for batch processing)
"""

from __future__ import annotations


import numpy as np

# Normalization ranges from DIQA-5000 val set (500 images, v1.0 predictions).
# Used for absolute mode when dataset statistics aren't available.
SIGMA_SQ_RANGE = (0.002, 0.019)  # (min, max) observed
ENTROPY_RANGE = (0.0, 0.693)  # (0, ln(5)) theoretical max for 5-class


def confidence_weight_absolute(
    sigma_sq: float,
    entropy: float,
    min_weight: float = 0.3,
    sigma_sq_range: tuple[float, float] = SIGMA_SQ_RANGE,
    entropy_range: tuple[float, float] = ENTROPY_RANGE,
) -> float:
    """Compute confidence weight from sigma_sq and entropy using fixed ranges.

    Normalizes each signal to [0, 1] using provided ranges, averages them,
    and maps to [min_weight, 1.0] via linear decay.

    Args:
        sigma_sq: Predicted variance from GaussianNLL head.
        entropy: Entropy of discretized level_probs distribution.
        min_weight: Floor for the confidence weight.
        sigma_sq_range: (min, max) for sigma_sq normalization.
        entropy_range: (min, max) for entropy normalization.

    Returns:
        Confidence weight in [min_weight, 1.0].
    """
    s_lo, s_hi = sigma_sq_range
    e_lo, e_hi = entropy_range

    s_norm = _clip_normalize(sigma_sq, s_lo, s_hi)
    e_norm = _clip_normalize(entropy, e_lo, e_hi)

    combined = (s_norm + e_norm) / 2.0
    return float(1.0 - (1.0 - min_weight) * combined)


def confidence_weights_rank(
    sigma_sqs: np.ndarray,
    entropies: np.ndarray,
    min_weight: float = 0.3,
) -> np.ndarray:
    """Compute confidence weights using rank-average over a batch.

    Rank-normalizes each signal to [0, 1] across the dataset, averages,
    and maps to [min_weight, 1.0]. More robust than absolute mode when
    signal distributions shift across datasets.

    Args:
        sigma_sqs: Array of shape (N,) with predicted variances.
        entropies: Array of shape (N,) with entropy values.
        min_weight: Floor for the confidence weight.

    Returns:
        Array of shape (N,) with weights in [min_weight, 1.0].
    """
    from scipy.stats import rankdata

    n = len(sigma_sqs)
    if n <= 1:
        return np.ones(n)

    s_ranks = (rankdata(sigma_sqs) - 1) / (n - 1)
    e_ranks = (rankdata(entropies) - 1) / (n - 1)

    combined = (s_ranks + e_ranks) / 2.0
    return 1.0 - (1.0 - min_weight) * combined


def _clip_normalize(value: float, lo: float, hi: float) -> float:
    """Normalize value to [0, 1] given range, with clipping."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))
