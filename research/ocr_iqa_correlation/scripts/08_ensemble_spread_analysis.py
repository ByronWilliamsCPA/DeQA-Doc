#!/usr/bin/env python3
"""Step 8: Multi-engine CER ensemble and inter-engine spread analysis.

Computes:
1. Ensemble CER (mean across engines) and its correlation with DeQA MOS
2. Subset ensembles (traditional-only, neural-only, all-excluding-VLM)
3. Inter-engine CER spread (std across engines) per image
4. Spread as a diagnostic signal — correlation with MOS, per-tier patterns
5. Engine-specific z-score normalized CER and ensemble correlation

Usage:
    python -m research.ocr_iqa_correlation.scripts.08_ensemble_spread_analysis
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Engine groupings based on architecture
ENGINE_GROUPS = {
    "traditional": ["tesseract", "easyocr", "rapidocr"],
    "neural": ["paddleocr", "doctr"],
    "cloud": ["gcloud_vision"],
    "vlm": ["glm-ocr", "deepseek-ocr2"],
    "traditional+neural": ["tesseract", "easyocr", "rapidocr", "paddleocr", "doctr"],
    "non_vlm": [
        "tesseract", "easyocr", "rapidocr", "paddleocr", "doctr",
        "gcloud_vision", "kraken",
    ],
    "top4_correlated": ["paddleocr", "tesseract", "easyocr", "doctr"],
    "all": [
        "tesseract", "easyocr", "rapidocr", "paddleocr", "doctr",
        "gcloud_vision", "kraken", "glm-ocr", "deepseek-ocr2",
    ],
}


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _compute_srcc(x: list[float], y: list[float]) -> tuple[float, float, int]:
    """Compute SRCC with p-value, filtering NaN/inf."""
    xa = np.array(x)
    ya = np.array(y)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[valid]
    ya = ya[valid]
    if len(xa) < 3:
        return float("nan"), float("nan"), int(len(xa))
    result = stats.spearmanr(xa, ya)
    return float(result.statistic), float(result.pvalue), int(len(xa))


def _compute_plcc(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute PLCC with p-value, filtering NaN/inf."""
    xa = np.array(x)
    ya = np.array(y)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[valid]
    ya = ya[valid]
    if len(xa) < 3:
        return float("nan"), float("nan")
    result = stats.pearsonr(xa, ya)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    """Run ensemble and spread analysis."""
    from research.ocr_iqa_correlation.config import (
        DATA_DIR,
        OUTPUTS_DIR,
    )

    # Load master dataset
    dataset_path = DATA_DIR / "dataset.jsonl"
    logger.info("Loading master dataset from %s", dataset_path)
    records = _load_jsonl(dataset_path)
    logger.info("Loaded %d records", len(records))

    # Discover available engines
    all_engines_in_data = set()
    for r in records:
        all_engines_in_data.update(r.get("ocr", {}).keys())
    logger.info("Engines in dataset: %s", sorted(all_engines_in_data))

    # ── Part 1: Ensemble CER ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 1: Ensemble CER analysis")

    ensemble_results: dict[str, dict] = {}

    for group_name, group_engines in ENGINE_GROUPS.items():
        # Filter to engines that exist in the data
        available = [e for e in group_engines if e in all_engines_in_data]
        if len(available) < 2:
            continue

        ensemble_cers: list[float] = []
        mos_values: list[float] = []
        per_image_data: list[dict] = []

        for record in records:
            mos = record.get("deqa_mos")
            if mos is None:
                continue

            ocr = record.get("ocr", {})
            engine_cers = []
            for engine in available:
                engine_data = ocr.get(engine)
                if engine_data and engine_data.get("cer") is not None:
                    engine_cers.append(engine_data["cer"])

            if len(engine_cers) < 2:
                continue

            mean_cer = float(np.mean(engine_cers))
            ensemble_cers.append(mean_cer)
            mos_values.append(mos)

            per_image_data.append({
                "image_id": record["image_id"],
                "tier": record["tier"],
                "ensemble_cer": round(mean_cer, 6),
                "n_engines": len(engine_cers),
                "deqa_mos": mos,
            })

        srcc, srcc_p, n = _compute_srcc(ensemble_cers, mos_values)
        plcc, plcc_p = _compute_plcc(ensemble_cers, mos_values)

        ensemble_results[group_name] = {
            "engines": available,
            "n_engines": len(available),
            "srcc": round(srcc, 4),
            "srcc_pvalue": srcc_p,
            "plcc": round(plcc, 4),
            "plcc_pvalue": plcc_p,
            "n_samples": n,
            "mean_ensemble_cer": round(float(np.mean(ensemble_cers)), 4),
        }

        logger.info(
            "  %-25s (%d engines): SRCC=%.4f (p=%.2e), PLCC=%.4f, n=%d",
            group_name, len(available), srcc, srcc_p, plcc, n,
        )

    # ── Part 2: Inter-engine CER spread ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 2: Inter-engine CER spread analysis")

    # Use non-VLM engines for spread (VLM hallucination inflates variance)
    spread_engines = [
        e for e in ENGINE_GROUPS["non_vlm"] if e in all_engines_in_data
    ]
    logger.info("Spread engines: %s", spread_engines)

    spread_data: list[dict] = []
    spread_values: list[float] = []
    mean_cer_values: list[float] = []
    mos_for_spread: list[float] = []

    for record in records:
        mos = record.get("deqa_mos")
        if mos is None:
            continue

        ocr = record.get("ocr", {})
        engine_cers = []
        for engine in spread_engines:
            engine_data = ocr.get(engine)
            if engine_data and engine_data.get("cer") is not None:
                engine_cers.append(engine_data["cer"])

        if len(engine_cers) < 3:
            continue

        mean_cer = float(np.mean(engine_cers))
        std_cer = float(np.std(engine_cers))
        cv_cer = std_cer / mean_cer if mean_cer > 0 else 0.0

        spread_values.append(std_cer)
        mean_cer_values.append(mean_cer)
        mos_for_spread.append(mos)

        spread_data.append({
            "image_id": record["image_id"],
            "tier": record["tier"],
            "mean_cer": round(mean_cer, 6),
            "std_cer": round(std_cer, 6),
            "cv_cer": round(cv_cer, 4),
            "deqa_mos": mos,
        })

    # Spread vs MOS correlation
    spread_srcc, spread_srcc_p, spread_n = _compute_srcc(spread_values, mos_for_spread)
    spread_plcc, spread_plcc_p = _compute_plcc(spread_values, mos_for_spread)

    logger.info(
        "  Spread(std) vs MOS: SRCC=%.4f (p=%.2e), PLCC=%.4f, n=%d",
        spread_srcc, spread_srcc_p, spread_plcc, spread_n,
    )

    # Spread vs ensemble CER correlation (is spread just a proxy for CER?)
    spread_cer_srcc, spread_cer_p, _ = _compute_srcc(spread_values, mean_cer_values)
    logger.info(
        "  Spread(std) vs mean CER: SRCC=%.4f (p=%.2e)",
        spread_cer_srcc, spread_cer_p,
    )

    # Per-tier spread statistics
    tier_spread: dict[str, dict] = defaultdict(lambda: {"spreads": [], "mean_cers": []})
    for item in spread_data:
        tier_spread[item["tier"]]["spreads"].append(item["std_cer"])
        tier_spread[item["tier"]]["mean_cers"].append(item["mean_cer"])

    per_tier_spread = {}
    for tier in ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]:
        if tier in tier_spread:
            spreads = tier_spread[tier]["spreads"]
            mean_cers = tier_spread[tier]["mean_cers"]
            per_tier_spread[tier] = {
                "mean_spread": round(float(np.mean(spreads)), 4),
                "std_spread": round(float(np.std(spreads)), 4),
                "mean_cer": round(float(np.mean(mean_cers)), 4),
                "n": len(spreads),
            }
            logger.info(
                "  Tier %-10s: mean_spread=%.4f, mean_cer=%.4f, n=%d",
                tier, np.mean(spreads), np.mean(mean_cers), len(spreads),
            )

    spread_results = {
        "engines": spread_engines,
        "n_engines": len(spread_engines),
        "spread_vs_mos": {
            "srcc": round(spread_srcc, 4),
            "srcc_pvalue": spread_srcc_p,
            "plcc": round(spread_plcc, 4),
            "plcc_pvalue": spread_plcc_p,
            "n": spread_n,
        },
        "spread_vs_mean_cer": {
            "srcc": round(spread_cer_srcc, 4),
            "srcc_pvalue": spread_cer_p,
        },
        "per_tier": per_tier_spread,
    }

    # ── Part 3: Z-score normalized CER ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 3: Z-score normalized CER ensemble")

    # Compute per-engine mean and std across all images
    engine_stats: dict[str, dict] = {}
    for engine in spread_engines:
        cers = []
        for record in records:
            ocr = record.get("ocr", {}).get(engine)
            if ocr and ocr.get("cer") is not None:
                cers.append(ocr["cer"])
        if cers:
            engine_stats[engine] = {
                "mean": float(np.mean(cers)),
                "std": float(np.std(cers)),
            }

    # Compute z-scored ensemble CER
    zscore_ensemble_cers: list[float] = []
    zscore_mos_values: list[float] = []

    for record in records:
        mos = record.get("deqa_mos")
        if mos is None:
            continue

        ocr = record.get("ocr", {})
        z_scores = []
        for engine in spread_engines:
            engine_data = ocr.get(engine)
            if engine_data and engine_data.get("cer") is not None:
                es = engine_stats.get(engine)
                if es and es["std"] > 0:
                    z = (engine_data["cer"] - es["mean"]) / es["std"]
                    z_scores.append(z)

        if len(z_scores) < 3:
            continue

        zscore_ensemble_cers.append(float(np.mean(z_scores)))
        zscore_mos_values.append(mos)

    z_srcc, z_srcc_p, z_n = _compute_srcc(zscore_ensemble_cers, zscore_mos_values)
    z_plcc, z_plcc_p = _compute_plcc(zscore_ensemble_cers, zscore_mos_values)

    logger.info(
        "  Z-score ensemble (non-VLM): SRCC=%.4f (p=%.2e), PLCC=%.4f, n=%d",
        z_srcc, z_srcc_p, z_plcc, z_n,
    )

    zscore_results = {
        "engines": spread_engines,
        "srcc": round(z_srcc, 4),
        "srcc_pvalue": z_srcc_p,
        "plcc": round(z_plcc, 4),
        "plcc_pvalue": z_plcc_p,
        "n_samples": z_n,
    }

    # ── Part 4: Paired ensemble analysis ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Part 4: Paired ensemble analysis")

    # Group by image_id
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_image[record["image_id"]][record["tier"]] = record

    paired_ensemble_results: dict[str, dict] = {}

    for group_name in ["top4_correlated", "non_vlm", "all"]:
        available = [e for e in ENGINE_GROUPS[group_name] if e in all_engines_in_data]
        if len(available) < 2:
            continue

        delta_cers: list[float] = []
        delta_moss: list[float] = []

        for image_id, tier_records in by_image.items():
            original = tier_records.get("ORIGINAL")
            if not original or original.get("deqa_mos") is None:
                continue

            orig_mos = original["deqa_mos"]
            orig_cers = []
            for engine in available:
                od = original.get("ocr", {}).get(engine)
                if od and od.get("cer") is not None:
                    orig_cers.append(od["cer"])
            if len(orig_cers) < 2:
                continue
            orig_mean_cer = float(np.mean(orig_cers))

            for tier, record in tier_records.items():
                if tier == "ORIGINAL":
                    continue
                dist_mos = record.get("deqa_mos")
                if dist_mos is None:
                    continue

                dist_cers = []
                for engine in available:
                    dd = record.get("ocr", {}).get(engine)
                    if dd and dd.get("cer") is not None:
                        dist_cers.append(dd["cer"])
                if len(dist_cers) < 2:
                    continue

                dist_mean_cer = float(np.mean(dist_cers))
                delta_cers.append(dist_mean_cer - orig_mean_cer)
                delta_moss.append(dist_mos - orig_mos)

        p_srcc, p_srcc_p, p_n = _compute_srcc(delta_cers, delta_moss)
        p_plcc, p_plcc_p = _compute_plcc(delta_cers, delta_moss)

        paired_ensemble_results[group_name] = {
            "engines": available,
            "n_engines": len(available),
            "paired_srcc": round(p_srcc, 4),
            "paired_srcc_pvalue": p_srcc_p,
            "paired_plcc": round(p_plcc, 4),
            "paired_plcc_pvalue": p_plcc_p,
            "n_pairs": p_n,
        }

        logger.info(
            "  Paired %-20s (%d engines): SRCC=%.4f (p=%.2e), PLCC=%.4f, n=%d",
            group_name, len(available), p_srcc, p_srcc_p, p_plcc, p_n,
        )

    # ── Save results ──────────────────────────────────────────────────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "description": "Multi-engine CER ensemble and inter-engine spread analysis",
        "ensemble_correlations": ensemble_results,
        "spread_analysis": spread_results,
        "zscore_ensemble": zscore_results,
        "paired_ensemble": paired_ensemble_results,
        "best_single_engine": {
            "engine": "paddleocr",
            "srcc": -0.658,
            "note": "PP-OCRv5 — strongest single-engine SRCC",
        },
    }

    report_path = OUTPUTS_DIR / "ensemble_spread_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("Summary: Ensemble vs Best Single Engine")
    logger.info("-" * 72)
    logger.info("%-30s %8s %8s %8s %6s", "Configuration", "SRCC", "PLCC", "Paired", "n")
    logger.info("-" * 72)

    logger.info("%-30s %8s %8s %8s %6s",
                "PP-OCRv5 (best single)", "-0.658", "-0.624", "-0.750", "1200")

    for group_name, data in sorted(ensemble_results.items()):
        paired = paired_ensemble_results.get(group_name, {}).get("paired_srcc", "—")
        if isinstance(paired, float):
            paired = f"{paired:.4f}"
        logger.info(
            "%-30s %8.4f %8.4f %8s %6d",
            f"Ensemble: {group_name} ({data['n_engines']})",
            data["srcc"], data["plcc"], paired, data["n_samples"],
        )

    logger.info(
        "%-30s %8.4f %8.4f %8s %6d",
        "Z-score ensemble (non-VLM)",
        zscore_results["srcc"], zscore_results["plcc"], "—", zscore_results["n_samples"],
    )

    logger.info("-" * 72)
    logger.info("Spread (std) vs MOS: SRCC=%.4f", spread_results["spread_vs_mos"]["srcc"])
    logger.info("Spread (std) vs mean CER: SRCC=%.4f", spread_results["spread_vs_mean_cer"]["srcc"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
