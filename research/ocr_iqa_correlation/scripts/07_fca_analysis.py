#!/usr/bin/env python3
"""Step 7: Compute FCA (Flexible Character Accuracy) from existing OCR results.

Reprocesses stored OCR text outputs against ground truth using line-level
alignment (FCA), then computes FCA-MOS correlations and compares with CER.

No OCR re-run needed — uses ocr_text from data/ocr_results/*.jsonl.

Usage:
    python -m research.ocr_iqa_correlation.scripts.07_fca_analysis
    python -m research.ocr_iqa_correlation.scripts.07_fca_analysis --engines tesseract paddleocr
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    """Compute FCA metrics and FCA-MOS correlations."""
    from research.ocr_iqa_correlation.analysis.cer_wer import (
        compute_cer,
        compute_fca,
    )
    from research.ocr_iqa_correlation.analysis.correlation import (
        compute_correlation,
    )
    from research.ocr_iqa_correlation.config import (
        DATA_DIR,
        GT_TEXT_DIR,
        OCR_RESULTS_DIR,
        OUTPUTS_DIR,
        SAMPLE_MANIFEST,
    )

    parser = argparse.ArgumentParser(description="Compute FCA from existing OCR results")
    parser.add_argument(
        "--engines",
        nargs="*",
        default=None,
        help="Specific engines to analyze (default: all available)",
    )
    args = parser.parse_args()

    # Load ground truth text
    logger.info("Loading ground truth text...")
    gt_texts: dict[str, str] = {}
    for gt_path in GT_TEXT_DIR.glob("*.txt"):
        image_id = gt_path.stem
        gt_texts[image_id] = gt_path.read_text(encoding="utf-8")
    logger.info("Loaded GT text for %d images", len(gt_texts))

    # Load DeQA scores
    deqa_path = DATA_DIR / "deqa_results" / "deqa_scores.jsonl"
    deqa_by_key: dict[str, float] = {}
    if deqa_path.exists():
        for r in _load_jsonl(deqa_path):
            key = f"{r['image_id']}_{r['tier']}"
            mos = r.get("deqa_mos") or r.get("deqa_overall_mos")
            if mos is not None:
                deqa_by_key[key] = float(mos)
    logger.info("Loaded %d DeQA scores", len(deqa_by_key))

    # Discover engines
    available_engines = sorted(
        p.stem for p in OCR_RESULTS_DIR.glob("*.jsonl")
    )
    engines = args.engines if args.engines else available_engines
    engines = [e for e in engines if e in available_engines]
    logger.info("Analyzing engines: %s", engines)

    # Compute FCA and CER per engine per image
    fca_results: dict[str, list[dict]] = {}

    for engine in engines:
        logger.info("Processing %s...", engine)
        ocr_records = _load_jsonl(OCR_RESULTS_DIR / f"{engine}.jsonl")
        engine_results = []

        for record in ocr_records:
            image_id = record["image_id"]
            tier = record["tier"]
            ocr_text = record.get("ocr_text", "")
            gt_text = gt_texts.get(image_id, "")

            if not gt_text:
                continue

            key = f"{image_id}_{tier}"
            mos = deqa_by_key.get(key)

            cer = compute_cer(gt_text, ocr_text)
            fca = compute_fca(gt_text, ocr_text)

            engine_results.append({
                "image_id": image_id,
                "tier": tier,
                "cer": round(cer, 6),
                "fca": round(fca, 6),
                "cer_minus_fca": round(cer - fca, 6),
                "deqa_mos": mos,
            })

        fca_results[engine] = engine_results
        logger.info(
            "  %s: %d images processed", engine, len(engine_results)
        )

    # Compute correlations: FCA vs MOS and CER vs MOS
    logger.info("Computing correlations...")
    correlation_comparison: dict[str, dict] = {}

    for engine in engines:
        records = fca_results[engine]
        records_with_mos = [r for r in records if r["deqa_mos"] is not None]

        if len(records_with_mos) < 10:
            logger.warning("  %s: too few records with MOS (%d)", engine, len(records_with_mos))
            continue

        cer_values = [r["cer"] for r in records_with_mos]
        fca_values = [r["fca"] for r in records_with_mos]
        mos_values = [r["deqa_mos"] for r in records_with_mos]

        cer_corr = compute_correlation(cer_values, mos_values)
        fca_corr = compute_correlation(fca_values, mos_values)

        # Per-tier mean FCA vs CER
        tier_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"cer_sum": 0, "fca_sum": 0, "n": 0})
        for r in records:
            tier_stats[r["tier"]]["cer_sum"] += r["cer"]
            tier_stats[r["tier"]]["fca_sum"] += r["fca"]
            tier_stats[r["tier"]]["n"] += 1

        per_tier = {}
        for tier, stats in sorted(tier_stats.items()):
            n = stats["n"]
            per_tier[tier] = {
                "mean_cer": round(stats["cer_sum"] / n, 4),
                "mean_fca": round(stats["fca_sum"] / n, 4),
                "mean_diff": round((stats["cer_sum"] - stats["fca_sum"]) / n, 4),
                "n": n,
            }

        mean_cer_minus_fca = sum(r["cer_minus_fca"] for r in records) / len(records)

        correlation_comparison[engine] = {
            "cer_srcc": round(cer_corr.srcc, 4),
            "cer_plcc": round(cer_corr.plcc, 4),
            "fca_srcc": round(fca_corr.srcc, 4),
            "fca_plcc": round(fca_corr.plcc, 4),
            "srcc_improvement": round(abs(fca_corr.srcc) - abs(cer_corr.srcc), 4),
            "mean_cer_minus_fca": round(mean_cer_minus_fca, 4),
            "n_samples": fca_corr.n_samples,
            "per_tier": per_tier,
        }

        logger.info(
            "  %s: CER SRCC=%.4f, FCA SRCC=%.4f, improvement=%.4f",
            engine,
            cer_corr.srcc,
            fca_corr.srcc,
            abs(fca_corr.srcc) - abs(cer_corr.srcc),
        )

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Per-image FCA results
    fca_output_path = OUTPUTS_DIR / "fca_per_image.jsonl"
    with open(fca_output_path, "w") as f:
        for engine in engines:
            for record in fca_results.get(engine, []):
                record_with_engine = {"engine": engine, **record}
                f.write(json.dumps(record_with_engine) + "\n")
    logger.info("Per-image FCA results: %s", fca_output_path)

    # Correlation comparison report
    report = {
        "description": "FCA vs CER correlation comparison with DeQA MOS",
        "engines": engines,
        "correlation_comparison": correlation_comparison,
    }

    report_path = OUTPUTS_DIR / "fca_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("FCA analysis report: %s", report_path)

    # Print summary table
    logger.info("=" * 72)
    logger.info("FCA vs CER Correlation Summary (SRCC with DeQA MOS)")
    logger.info("-" * 72)
    logger.info("%-20s %10s %10s %12s %12s", "Engine", "CER SRCC", "FCA SRCC", "Improvement", "Mean CER-FCA")
    logger.info("-" * 72)
    for engine, data in sorted(
        correlation_comparison.items(),
        key=lambda x: x[1]["fca_srcc"],
    ):
        logger.info(
            "%-20s %10.4f %10.4f %+12.4f %12.4f",
            engine,
            data["cer_srcc"],
            data["fca_srcc"],
            data["srcc_improvement"],
            data["mean_cer_minus_fca"],
        )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
