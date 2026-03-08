#!/usr/bin/env python3
"""Step 5: Analyze results — CER/WER, correlations, paired analysis, plots.

Merges OCR results, DeQA scores, and GT text into a master dataset,
then computes correlations and generates visualizations.

Usage:
    python -m research.ocr_iqa_correlation.scripts.05_analyze
"""

from __future__ import annotations

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
            records.append(json.loads(line))
    return records


def _build_master_dataset(
    samples: list[dict],
    distortion_records: list[dict],
    ocr_records: dict[str, list[dict]],
    deqa_records: list[dict],
    gt_text_dir: Path,
) -> list[dict]:
    """Merge all data sources into a master dataset.

    Returns:
        List of master records with all fields populated.
    """
    from research.ocr_iqa_correlation.analysis.cer_wer import compute_metrics

    # Index GT text by image_id
    gt_texts = {}
    for sample in samples:
        gt_path = gt_text_dir / f"{sample['image_id']}.txt"
        if gt_path.exists():
            gt_texts[sample["image_id"]] = gt_path.read_text(encoding="utf-8")

    # Index distortion metadata
    dist_by_key = {}
    for r in distortion_records:
        key = f"{r['image_id']}_{r['tier']}"
        dist_by_key[key] = r

    # Index DeQA scores
    deqa_by_key = {}
    for r in deqa_records:
        key = f"{r['image_id']}_{r['tier']}"
        deqa_by_key[key] = r

    # Index OCR results by (image_id, tier, engine)
    ocr_by_key: dict[str, dict[str, dict]] = defaultdict(dict)
    for engine_name, records in ocr_records.items():
        for r in records:
            key = f"{r['image_id']}_{r['tier']}"
            ocr_by_key[key][engine_name] = r

    # Build master records
    master = []
    for dist_record in distortion_records:
        image_id = dist_record["image_id"]
        tier = dist_record["tier"]
        key = f"{image_id}_{tier}"

        gt_text = gt_texts.get(image_id, "")
        deqa = deqa_by_key.get(key, {})
        ocr_engines_data = ocr_by_key.get(key, {})

        # Compute CER/WER for each engine
        ocr_results = {}
        for engine_name, ocr_data in ocr_engines_data.items():
            ocr_text = ocr_data.get("ocr_text", "")
            metrics = compute_metrics(gt_text, ocr_text)
            ocr_results[engine_name] = {
                "cer": round(metrics["cer"], 6),
                "wer": round(metrics["wer"], 6),
                "chars": ocr_data.get("ocr_chars", len(ocr_text)),
                "time_ms": ocr_data.get("time_ms", 0),
            }

        record = {
            "image_id": image_id,
            "tier": tier,
            "source_dataset": next(
                (s["source_dataset"] for s in samples if s["image_id"] == image_id),
                "unknown",
            ),
            "image_path": dist_record.get("image_path", ""),
            "gt_text_chars": len(gt_text),
            "seed": dist_record.get("seed"),
            "actual_overall_quality": dist_record.get("iqa_labels", {}).get(
                "overall_quality", 1.0
            ),
            "iqa_labels": dist_record.get("iqa_labels", {}),
            "deqa_mos": deqa.get("deqa_mos"),
            "ocr": ocr_results,
        }
        master.append(record)

    return master


def main() -> None:
    """Run full analysis pipeline."""
    from research.ocr_iqa_correlation.analysis.correlation import (
        compute_per_engine_correlations,
        compute_per_tier_stats,
    )
    from research.ocr_iqa_correlation.analysis.paired_analysis import (
        compute_paired_correlations,
        compute_paired_deltas,
        compute_tier_significance,
    )
    from research.ocr_iqa_correlation.analysis.visualize import (
        plot_cer_boxplots,
        plot_cer_vs_mos_scatter,
        plot_engine_tier_heatmap,
        plot_paired_deltas,
    )
    from research.ocr_iqa_correlation.config import (
        DATA_DIR,
        DATASET_JSONL,
        GT_TEXT_DIR,
        OCR_RESULTS_DIR,
        OUTPUTS_DIR,
        SAMPLE_MANIFEST,
    )

    # Load all data
    logger.info("Loading data sources...")

    with open(SAMPLE_MANIFEST) as f:
        samples = json.load(f)

    distortion_records = _load_jsonl(DATA_DIR / "distortion_metadata.jsonl")

    # Load OCR results (one file per engine)
    ocr_records = {}
    for ocr_path in OCR_RESULTS_DIR.glob("*.jsonl"):
        engine_name = ocr_path.stem
        ocr_records[engine_name] = _load_jsonl(ocr_path)

    # Load DeQA scores
    deqa_path = DATA_DIR / "deqa_results" / "deqa_scores.jsonl"
    deqa_records = _load_jsonl(deqa_path) if deqa_path.exists() else []

    engines = sorted(ocr_records.keys())
    logger.info("Engines found: %s", engines)
    logger.info("DeQA records: %d", len(deqa_records))

    # Build master dataset
    logger.info("Building master dataset...")
    master = _build_master_dataset(
        samples, distortion_records, ocr_records, deqa_records, GT_TEXT_DIR
    )

    # Save master dataset
    with open(DATASET_JSONL, "w") as f:
        for record in master:
            f.write(json.dumps(record) + "\n")
    logger.info("Master dataset: %s (%d records)", DATASET_JSONL, len(master))

    # Skip correlation analysis if no DeQA scores
    if not deqa_records:
        logger.warning("No DeQA scores found — skipping correlation analysis.")
        logger.warning("Run step 04 first, then re-run this script.")

    # ── Correlation Analysis ──
    if deqa_records:
        logger.info("Computing per-engine correlations (CER ↔ MOS)...")
        engine_correlations = compute_per_engine_correlations(master, engines)

        logger.info("Computing per-tier statistics...")
        tier_stats = compute_per_tier_stats(master, engines)

        logger.info("Computing paired correlations (ΔCER ↔ ΔMOS)...")
        paired_correlations = compute_paired_correlations(master, engines)

        logger.info("Computing tier significance tests...")
        tier_significance = compute_tier_significance(master, engines)

    # ── Visualization ──
    logger.info("Generating visualizations...")

    for engine in engines:
        plot_cer_vs_mos_scatter(master, engine)

    plot_cer_boxplots(master, engines)

    if deqa_records:
        plot_engine_tier_heatmap(tier_stats, engines)

        deltas = compute_paired_deltas(master, engines)
        plot_paired_deltas(deltas, engines)

    # ── Summary Report ──
    report_path = OUTPUTS_DIR / "correlation_report.json"
    report = {
        "n_base_images": len(samples),
        "n_total_images": len(master),
        "engines": engines,
        "tiers": sorted({r["tier"] for r in master}),
    }

    if deqa_records:
        report["engine_correlations"] = {
            engine: {
                "srcc": result.srcc,
                "srcc_pvalue": result.srcc_pvalue,
                "plcc": result.plcc,
                "plcc_pvalue": result.plcc_pvalue,
                "n_samples": result.n_samples,
            }
            for engine, result in engine_correlations.items()
        }
        report["paired_correlations"] = {
            engine: {
                "srcc": result.srcc,
                "srcc_pvalue": result.srcc_pvalue,
                "plcc": result.plcc,
                "plcc_pvalue": result.plcc_pvalue,
                "n_samples": result.n_samples,
            }
            for engine, result in paired_correlations.items()
        }
        report["tier_stats"] = tier_stats
        report["tier_significance"] = tier_significance

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Analysis complete:")
    logger.info("  Master dataset: %s", DATASET_JSONL)
    logger.info("  Report: %s", report_path)

    if deqa_records:
        logger.info("  Correlations (CER ↔ MOS):")
        for engine, result in engine_correlations.items():
            logger.info(
                "    %s: SRCC=%.4f (p=%.2e), PLCC=%.4f (p=%.2e)",
                engine, result.srcc, result.srcc_pvalue, result.plcc, result.plcc_pvalue,
            )


if __name__ == "__main__":
    main()
