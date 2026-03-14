"""Correlation metrics for VLM vs human MOS comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_correlations(
    vlm_scores: list[float],
    human_mos: list[float],
) -> dict[str, Any]:
    """Compute SRCC and PLCC between VLM scores and human MOS.

    Args:
        vlm_scores: VLM-predicted quality scores.
        human_mos: Human Mean Opinion Scores (ground truth).

    Returns:
        Dictionary with srcc, plcc, p-values, and distribution statistics.
    """
    vlm_arr = np.array(vlm_scores)
    mos_arr = np.array(human_mos)

    # Remove NaN entries
    valid = ~(np.isnan(vlm_arr) | np.isnan(mos_arr))
    vlm_arr = vlm_arr[valid]
    mos_arr = mos_arr[valid]

    if len(vlm_arr) < 3:
        return {"srcc": 0.0, "plcc": 0.0, "n": int(len(vlm_arr))}

    srcc, srcc_p = spearmanr(vlm_arr, mos_arr)
    plcc, plcc_p = pearsonr(vlm_arr, mos_arr)

    return {
        "srcc": float(srcc),
        "srcc_pvalue": float(srcc_p),
        "plcc": float(plcc),
        "plcc_pvalue": float(plcc_p),
        "n": int(len(vlm_arr)),
        "vlm_mean": float(vlm_arr.mean()),
        "vlm_std": float(vlm_arr.std()),
        "vlm_min": float(vlm_arr.min()),
        "vlm_max": float(vlm_arr.max()),
        "mos_mean": float(mos_arr.mean()),
        "mos_std": float(mos_arr.std()),
    }
