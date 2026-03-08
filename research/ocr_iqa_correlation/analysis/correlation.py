"""Correlation analysis between OCR error rates and IQA scores.

Computes Spearman Rank-Order Correlation (SRCC) and Pearson Linear
Correlation (PLCC) between CER/WER and DeQA MOS scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationResult:
    """Result of a correlation computation.

    Attributes:
        srcc: Spearman rank-order correlation coefficient.
        srcc_pvalue: P-value for SRCC.
        plcc: Pearson linear correlation coefficient.
        plcc_pvalue: P-value for PLCC.
        n_samples: Number of samples used.
    """

    srcc: float
    srcc_pvalue: float
    plcc: float
    plcc_pvalue: float
    n_samples: int


def compute_correlation(
    error_rates: list[float],
    quality_scores: list[float],
) -> CorrelationResult:
    """Compute SRCC and PLCC between error rates and quality scores.

    Expects negative correlation: higher quality → lower error rate.

    Args:
        error_rates: CER or WER values per image.
        quality_scores: DeQA MOS scores per image.

    Returns:
        CorrelationResult with SRCC, PLCC, p-values, and sample count.
    """
    x = np.array(error_rates)
    y = np.array(quality_scores)

    # Filter out NaN/inf
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        logger.warning("Too few valid samples (%d) for correlation", len(x))
        return CorrelationResult(
            srcc=float("nan"),
            srcc_pvalue=float("nan"),
            plcc=float("nan"),
            plcc_pvalue=float("nan"),
            n_samples=len(x),
        )

    srcc_result = stats.spearmanr(x, y)
    plcc_result = stats.pearsonr(x, y)

    return CorrelationResult(
        srcc=float(srcc_result.statistic),
        srcc_pvalue=float(srcc_result.pvalue),
        plcc=float(plcc_result.statistic),
        plcc_pvalue=float(plcc_result.pvalue),
        n_samples=len(x),
    )


def compute_per_engine_correlations(
    dataset_records: list[dict],
    engines: list[str],
) -> dict[str, CorrelationResult]:
    """Compute CER↔MOS correlation per OCR engine.

    Args:
        dataset_records: Master dataset records with ocr and deqa_mos fields.
        engines: List of engine names to analyze.

    Returns:
        Dict mapping engine name to CorrelationResult.
    """
    results = {}

    for engine in engines:
        cer_values = []
        mos_values = []

        for record in dataset_records:
            ocr_data = record.get("ocr", {}).get(engine)
            mos = record.get("deqa_mos")

            if ocr_data is None or mos is None:
                continue

            cer = ocr_data.get("cer")
            if cer is not None:
                cer_values.append(cer)
                mos_values.append(mos)

        result = compute_correlation(cer_values, mos_values)
        results[engine] = result

        logger.info(
            "Engine %s: SRCC=%.4f (p=%.4g), PLCC=%.4f (p=%.4g), n=%d",
            engine,
            result.srcc,
            result.srcc_pvalue,
            result.plcc,
            result.plcc_pvalue,
            result.n_samples,
        )

    return results


def compute_per_tier_stats(
    dataset_records: list[dict],
    engines: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-tier mean CER and MOS with confidence intervals.

    Args:
        dataset_records: Master dataset records.
        engines: List of engine names.

    Returns:
        Nested dict: tier -> engine -> {mean_cer, std_cer, mean_mos, std_mos, n}.
    """
    from collections import defaultdict

    tier_data: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in dataset_records:
        tier = record.get("tier", "UNKNOWN")
        for engine in engines:
            ocr_data = record.get("ocr", {}).get(engine)
            mos = record.get("deqa_mos")
            if ocr_data and mos is not None:
                tier_data[tier][engine].append({
                    "cer": ocr_data.get("cer", 0.0),
                    "mos": mos,
                })

    results: dict[str, dict[str, dict[str, float]]] = {}
    for tier, engine_data in sorted(tier_data.items()):
        results[tier] = {}
        for engine, records in engine_data.items():
            cers = [r["cer"] for r in records]
            moss = [r["mos"] for r in records]
            results[tier][engine] = {
                "mean_cer": float(np.mean(cers)),
                "std_cer": float(np.std(cers)),
                "mean_mos": float(np.mean(moss)),
                "std_mos": float(np.std(moss)),
                "n": len(records),
            }

    return results
