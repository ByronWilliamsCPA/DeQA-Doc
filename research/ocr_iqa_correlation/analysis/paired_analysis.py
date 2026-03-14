"""Paired analysis comparing base images to their distorted versions.

Computes delta metrics (CER change, MOS change) between original and
distorted versions, then correlates the deltas. More robust than
absolute value correlation since it controls for per-image baseline.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy import stats

from research.ocr_iqa_correlation.analysis.correlation import (
    CorrelationResult,
    compute_correlation,
)

logger = logging.getLogger(__name__)


def compute_paired_deltas(
    dataset_records: list[dict],
    engines: list[str],
) -> dict[str, list[dict[str, float]]]:
    """Compute delta CER and delta MOS for each base↔distorted pair.

    For each (image_id, engine), computes:
        delta_cer = CER(distorted) - CER(original)
        delta_mos = MOS(distorted) - MOS(original)

    Args:
        dataset_records: Master dataset records.
        engines: List of engine names.

    Returns:
        Dict mapping engine name to list of delta records.
    """
    # Group records by image_id
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in dataset_records:
        image_id = record["image_id"]
        tier = record["tier"]
        by_image[image_id][tier] = record

    results: dict[str, list[dict[str, float]]] = defaultdict(list)

    for image_id, tier_records in by_image.items():
        original = tier_records.get("ORIGINAL")
        if original is None:
            continue

        original_mos = original.get("deqa_mos")
        if original_mos is None:
            continue

        for tier, record in tier_records.items():
            if tier == "ORIGINAL":
                continue

            distorted_mos = record.get("deqa_mos")
            if distorted_mos is None:
                continue

            for engine in engines:
                orig_ocr = original.get("ocr", {}).get(engine)
                dist_ocr = record.get("ocr", {}).get(engine)

                if orig_ocr is None or dist_ocr is None:
                    continue

                orig_cer = orig_ocr.get("cer")
                dist_cer = dist_ocr.get("cer")

                if orig_cer is None or dist_cer is None:
                    continue

                results[engine].append({
                    "image_id": image_id,
                    "tier": tier,
                    "delta_cer": dist_cer - orig_cer,
                    "delta_mos": distorted_mos - original_mos,
                    "original_cer": orig_cer,
                    "distorted_cer": dist_cer,
                    "original_mos": original_mos,
                    "distorted_mos": distorted_mos,
                })

    return dict(results)


def compute_paired_correlations(
    dataset_records: list[dict],
    engines: list[str],
) -> dict[str, CorrelationResult]:
    """Compute correlation between delta CER and delta MOS.

    Args:
        dataset_records: Master dataset records.
        engines: List of engine names.

    Returns:
        Dict mapping engine name to CorrelationResult for the deltas.
    """
    deltas = compute_paired_deltas(dataset_records, engines)
    results = {}

    for engine, delta_records in deltas.items():
        delta_cers = [d["delta_cer"] for d in delta_records]
        delta_moss = [d["delta_mos"] for d in delta_records]

        result = compute_correlation(delta_cers, delta_moss)
        results[engine] = result

        logger.info(
            "Paired %s: SRCC=%.4f (p=%.4g), PLCC=%.4f (p=%.4g), n=%d",
            engine,
            result.srcc,
            result.srcc_pvalue,
            result.plcc,
            result.plcc_pvalue,
            result.n_samples,
        )

    return results


def compute_tier_significance(
    dataset_records: list[dict],
    engines: list[str],
    tier_order: list[str] | None = None,
) -> dict[str, list[dict]]:
    """Test significance of CER differences between adjacent tiers.

    Uses Wilcoxon signed-rank test on paired samples (same base image,
    different tiers).

    Args:
        dataset_records: Master dataset records.
        engines: List of engine names.
        tier_order: Ordered list of tier names. Defaults to
            ORIGINAL, PRISTINE, HIGH, MEDIUM, LOW, DEGRADED.

    Returns:
        Dict mapping engine name to list of pairwise test results.
    """
    if tier_order is None:
        tier_order = ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]

    # Group by image_id and tier
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in dataset_records:
        by_image[record["image_id"]][record["tier"]] = record

    results: dict[str, list[dict]] = {}

    for engine in engines:
        pairwise_tests = []

        for i in range(len(tier_order) - 1):
            tier_a = tier_order[i]
            tier_b = tier_order[i + 1]

            cer_a = []
            cer_b = []

            for image_id, tiers in by_image.items():
                rec_a = tiers.get(tier_a)
                rec_b = tiers.get(tier_b)

                if rec_a is None or rec_b is None:
                    continue

                ocr_a = rec_a.get("ocr", {}).get(engine, {})
                ocr_b = rec_b.get("ocr", {}).get(engine, {})

                ca = ocr_a.get("cer")
                cb = ocr_b.get("cer")

                if ca is not None and cb is not None:
                    cer_a.append(ca)
                    cer_b.append(cb)

            if len(cer_a) < 5:
                pairwise_tests.append({
                    "tier_a": tier_a,
                    "tier_b": tier_b,
                    "n_pairs": len(cer_a),
                    "statistic": None,
                    "pvalue": None,
                    "mean_diff": None,
                })
                continue

            diff = np.array(cer_b) - np.array(cer_a)
            try:
                stat_result = stats.wilcoxon(diff, alternative="greater")
                pairwise_tests.append({
                    "tier_a": tier_a,
                    "tier_b": tier_b,
                    "n_pairs": len(cer_a),
                    "statistic": float(stat_result.statistic),
                    "pvalue": float(stat_result.pvalue),
                    "mean_diff": float(np.mean(diff)),
                })
            except ValueError:
                pairwise_tests.append({
                    "tier_a": tier_a,
                    "tier_b": tier_b,
                    "n_pairs": len(cer_a),
                    "statistic": None,
                    "pvalue": None,
                    "mean_diff": float(np.mean(diff)),
                })

        results[engine] = pairwise_tests

    return results
