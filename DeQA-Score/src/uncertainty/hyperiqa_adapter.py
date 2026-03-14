"""Adapter for HyperIQA++ 10-bin predictions to 5-level DeQA format.

HyperIQA++ outputs 10-bin soft-label distributions per dimension with
bin centers uniformly spaced in [1.0, 5.0] (width=0.4). This module
converts those to the 5-level DeQA convention [excellent, good, fair,
poor, bad] and wraps them in a CrossValidator for use as an independent
uncertainty signal in the fusion pipeline.

10-bin to 5-level mapping (nearest level assignment):
    excellent (score 5, idx 0): bin 9  (center 4.8)
    good      (score 4, idx 1): bins 6,7,8  (centers 3.6, 4.0, 4.4)
    fair      (score 3, idx 2): bins 4,5  (centers 2.8, 3.2)
    poor      (score 2, idx 3): bins 1,2,3  (centers 1.6, 2.0, 2.4)
    bad       (score 1, idx 4): bin 0  (center 1.2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_validator import CrossValidator
from .gaussian_to_discrete import _norm_cdf

logger = logging.getLogger(__name__)

# HyperIQA++ 10-bin centers uniformly spaced in [1.0, 5.0]
HYPERIQA_BIN_CENTERS = np.array(
    [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8],
    dtype=np.float64,
)

# Default mapping: 10-bin index → 5-level DeQA index
# Assigns each bin center to the nearest integer level (1-5),
# then maps to DeQA index order [excellent=0, ..., bad=4].
#   bin 0 (1.2) → level 1 (bad)     → idx 4
#   bin 1 (1.6) → level 2 (poor)    → idx 3
#   bin 2 (2.0) → level 2 (poor)    → idx 3
#   bin 3 (2.4) → level 2 (poor)    → idx 3
#   bin 4 (2.8) → level 3 (fair)    → idx 2
#   bin 5 (3.2) → level 3 (fair)    → idx 2
#   bin 6 (3.6) → level 4 (good)    → idx 1
#   bin 7 (4.0) → level 4 (good)    → idx 1
#   bin 8 (4.4) → level 4 (good)    → idx 1
#   bin 9 (4.8) → level 5 (excellent) → idx 0
DEFAULT_10_TO_5_MAP = np.array([4, 3, 3, 3, 2, 2, 1, 1, 1, 0], dtype=np.int64)

# HyperIQA++ dimension names → DeQA pipeline dimension names
DIMENSION_ALIASES: dict[str, str] = {
    "color_fidelity": "color",
    "colour": "color",
    "colour_fidelity": "color",
}


@dataclass(frozen=True)
class BinMapping:
    """Mapping from N-bin to 5-level distributions.

    Attributes:
        n_bins: Number of source bins.
        bin_to_level: Array of shape (n_bins,) mapping each bin index
            to a 5-level DeQA index (0=excellent, 4=bad).
        bin_centers: Array of shape (n_bins,) with bin center values.
    """

    n_bins: int
    bin_to_level: np.ndarray
    bin_centers: np.ndarray

    @classmethod
    def default_10bin(cls) -> BinMapping:
        """Standard HyperIQA++ 10-bin mapping."""
        return cls(
            n_bins=10,
            bin_to_level=DEFAULT_10_TO_5_MAP.copy(),
            bin_centers=HYPERIQA_BIN_CENTERS.copy(),
        )


def map_bins_to_levels(
    bin_probs: np.ndarray,
    mapping: BinMapping | None = None,
) -> np.ndarray:
    """Convert N-bin probability distribution to 5-level DeQA distribution.

    Sums bin probabilities that map to the same quality level.

    Args:
        bin_probs: Array of shape (N,) with probabilities for each bin.
        mapping: Bin-to-level mapping. Defaults to standard 10-bin mapping.

    Returns:
        Array of shape (5,) in DeQA convention [excellent→bad],
        normalized to sum to 1.0.

    Raises:
        ValueError: If bin_probs shape doesn't match mapping.n_bins.
    """
    if mapping is None:
        mapping = BinMapping.default_10bin()

    bin_probs = np.asarray(bin_probs, dtype=np.float64).ravel()
    if bin_probs.shape[0] != mapping.n_bins:
        msg = (
            f"Expected {mapping.n_bins}-element distribution, "
            f"got shape {bin_probs.shape}"
        )
        raise ValueError(msg)

    level_probs = np.zeros(5, dtype=np.float64)
    for bin_idx in range(mapping.n_bins):
        level_idx = mapping.bin_to_level[bin_idx]
        level_probs[level_idx] += bin_probs[bin_idx]

    total = level_probs.sum()
    if total > 0:
        level_probs /= total
    else:
        level_probs[:] = 0.2  # uniform fallback

    return level_probs


def gaussian_to_10bin(
    mu: float,
    sigma_sq: float,
    bin_centers: np.ndarray = HYPERIQA_BIN_CENTERS,
    bin_width: float = 0.4,
) -> np.ndarray:
    """Convert SigLIP2's (μ, σ²) to a 10-bin distribution matching HyperIQA++.

    Integrates a Gaussian N(μ, σ²) over each bin using CDF at bin edges.
    Bin edges are computed as center ± bin_width/2.

    Args:
        mu: Predicted quality score (clamped to [1.0, 5.0]).
        sigma_sq: Predicted variance from SigLIP2's GaussianNLL head.
        bin_centers: Array of 10 bin center values.
        bin_width: Width of each bin (uniform spacing).

    Returns:
        Array of shape (10,) with probabilities summing to 1.0.
    """
    mu = float(np.clip(mu, 1.0, 5.0))
    sigma = max(float(np.sqrt(sigma_sq)), 0.1)

    probs = np.empty(len(bin_centers), dtype=np.float64)
    for i, center in enumerate(bin_centers):
        lo = center - bin_width / 2.0
        hi = center + bin_width / 2.0
        probs[i] = _norm_cdf(hi, mu, sigma) - _norm_cdf(lo, mu, sigma)

    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        nearest = int(np.argmin(np.abs(bin_centers - mu)))
        probs[:] = 0.0
        probs[nearest] = 1.0

    return probs


def native_10bin_jsd(
    siglip2_mu: float,
    siglip2_sigma_sq: float,
    hyperiqa_10bin: np.ndarray,
    bin_centers: np.ndarray = HYPERIQA_BIN_CENTERS,
    bin_width: float = 0.4,
) -> float:
    """Compute JSD between SigLIP2 and HyperIQA++ in native 10-bin space.

    Avoids the lossy 10-bin→5-level mapping by converting SigLIP2's
    Gaussian prediction to a 10-bin distribution via CDF integration,
    then computing JSD directly against HyperIQA++'s native output.

    Args:
        siglip2_mu: SigLIP2's predicted MOS.
        siglip2_sigma_sq: SigLIP2's predicted variance.
        hyperiqa_10bin: HyperIQA++'s raw 10-bin probability distribution.
        bin_centers: HyperIQA++ bin center values.
        bin_width: Width of each bin.

    Returns:
        JSD in nats, bounded in [0, ln(2) ≈ 0.693].
    """
    siglip2_10bin = gaussian_to_10bin(
        siglip2_mu, siglip2_sigma_sq, bin_centers, bin_width,
    )
    q = np.asarray(hyperiqa_10bin, dtype=np.float64).ravel()
    # Compute JSD for arbitrary-length distributions (discrete_metrics.discrete_jsd
    # is hardcoded to 5-element; we need N-element here)
    eps = 1e-12
    p = np.maximum(siglip2_10bin, eps)
    q = np.maximum(q, eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * kl_pm + 0.5 * kl_qm


def _parse_hyperiqa_record(
    record: dict,
    image_id: str,
    mapping: BinMapping,
    result: dict[str, dict[str, np.ndarray]],
) -> bool:
    """Parse a single HyperIQA++ record into the result dict.

    Tries multi-dimension format first, then falls back to flat format.

    Args:
        record: Parsed JSON record from JSONL line.
        image_id: Image identifier extracted from the record.
        mapping: Bin-to-level mapping configuration.
        result: Accumulator dict to update in place.

    Returns:
        True if at least one dimension was extracted, False otherwise.
    """
    dims_found = False
    for dim_key in ("overall", "sharpness", "color", "color_fidelity"):
        if dim_key in record and isinstance(record[dim_key], dict):
            probs_raw = record[dim_key].get("probs")
            if probs_raw is None:
                continue
            dim_name = DIMENSION_ALIASES.get(dim_key, dim_key)
            bin_probs = np.array(probs_raw, dtype=np.float64)
            level_probs = map_bins_to_levels(bin_probs, mapping)
            result.setdefault(dim_name, {})[image_id] = level_probs
            dims_found = True

    if not dims_found and "probs" in record:
        dim_key = record.get("dimension", "overall")
        dim_name = DIMENSION_ALIASES.get(dim_key, dim_key)
        bin_probs = np.array(record["probs"], dtype=np.float64)
        level_probs = map_bins_to_levels(bin_probs, mapping)
        result.setdefault(dim_name, {})[image_id] = level_probs
        dims_found = True

    return dims_found


def load_hyperiqa_predictions(
    path: str | Path,
    mapping: BinMapping | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load HyperIQA++ JSONL predictions and convert to 5-level probs.

    Expected JSONL format per line:
        {
            "image": "path.jpg",
            "overall": {"score": 3.5, "probs": [0.01, ...10 elements...]},
            "sharpness": {"score": 3.2, "probs": [...]},
            "color": {"score": 3.8, "probs": [...]},
            ...
        }

    Also supports flat format:
        {
            "image": "path.jpg",
            "dimension": "overall",
            "probs": [0.01, ...10 elements...],
            "score": 3.5
        }

    Args:
        path: Path to JSONL file with HyperIQA++ predictions.
        mapping: Bin-to-level mapping. Defaults to standard 10-bin.

    Returns:
        Nested dict {dimension: {image_id: level_probs(5,)}}.
    """
    if mapping is None:
        mapping = BinMapping.default_10bin()

    path = Path(path)
    result: dict[str, dict[str, np.ndarray]] = {}
    n_loaded = 0
    n_skipped = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            image_id = record.get("image", record.get("id", ""))
            if not image_id:
                n_skipped += 1
                continue

            if _parse_hyperiqa_record(record, image_id, mapping, result):
                n_loaded += 1
            else:
                n_skipped += 1

    logger.info(
        "Loaded HyperIQA++ predictions: %d images, %d skipped, dims=%s",
        n_loaded,
        n_skipped,
        list(result.keys()),
    )
    return result


def hyperiqa_cross_validator(
    path: str | Path,
    mapping: BinMapping | None = None,
) -> CrossValidator:
    """Create a CrossValidator from HyperIQA++ prediction file.

    Loads 10-bin predictions, converts to 5-level, and wraps in the
    standard CrossValidator interface so it can be used as a drop-in
    second cross-validator alongside the DeQA one.

    Args:
        path: Path to HyperIQA++ JSONL predictions.
        mapping: Bin-to-level mapping. Defaults to standard 10-bin.

    Returns:
        CrossValidator instance using HyperIQA++ predictions.
    """
    level_probs = load_hyperiqa_predictions(path, mapping)
    return CrossValidator.from_level_probs_dict(level_probs)
