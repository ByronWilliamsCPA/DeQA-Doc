"""Batch OCR runner with checkpoint/resume support.

Processes all images across all engines, saving results incrementally
to per-engine JSONL files. Skips already-processed image+engine pairs
on resume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research.ocr_iqa_correlation.config import OCR_RESULTS_DIR
from research.ocr_iqa_correlation.ocr.base import OCREngine, OCRResult

logger = logging.getLogger(__name__)


def _load_completed(results_path: Path) -> set[str]:
    """Load set of already-processed image_ids from a JSONL file."""
    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                record = json.loads(line)
                key = f"{record['image_id']}_{record['tier']}"
                completed.add(key)
    return completed


def run_ocr_engine(
    engine: OCREngine,
    image_records: list[dict],
    output_dir: Path = OCR_RESULTS_DIR,
) -> Path:
    """Run a single OCR engine on all images with resume support.

    Args:
        engine: OCR engine instance.
        image_records: List of dicts with image_id, tier, image_path.
        output_dir: Directory for per-engine JSONL output.

    Returns:
        Path to the output JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{engine.name}.jsonl"

    completed = _load_completed(results_path)
    remaining = [
        r
        for r in image_records
        if f"{r['image_id']}_{r['tier']}" not in completed
    ]

    logger.info(
        "Engine %s: %d total, %d completed, %d remaining",
        engine.name,
        len(image_records),
        len(completed),
        len(remaining),
    )

    with open(results_path, "a") as f:
        for idx, record in enumerate(remaining):
            image_id = record["image_id"]
            tier = record["tier"]
            image_path = record["image_path"]

            if (idx + 1) % 50 == 0:
                logger.info(
                    "Engine %s: processing %d/%d", engine.name, idx + 1, len(remaining)
                )

            result = engine.recognize(image_path)

            output_record = {
                "image_id": image_id,
                "tier": tier,
                "engine": engine.name,
                "ocr_text": result.text,
                "ocr_chars": len(result.text),
                "time_ms": round(result.time_ms, 1),
                "error": result.error,
            }

            f.write(json.dumps(output_record) + "\n")
            f.flush()

    logger.info("Engine %s: complete. Results at %s", engine.name, results_path)
    return results_path


def run_all_engines(
    engines: list[OCREngine],
    image_records: list[dict],
    output_dir: Path = OCR_RESULTS_DIR,
) -> dict[str, Path]:
    """Run all OCR engines on all images.

    Args:
        engines: List of OCR engine instances.
        image_records: List of dicts with image_id, tier, image_path.
        output_dir: Directory for per-engine JSONL output.

    Returns:
        Dict mapping engine name to output JSONL path.
    """
    results = {}
    for engine in engines:
        logger.info("Starting OCR engine: %s", engine.name)
        path = run_ocr_engine(engine, image_records, output_dir)
        results[engine.name] = path

    return results
