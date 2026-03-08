#!/usr/bin/env python3
"""Step 3: Run OCR engines on all images.

Runs 4 OCR engines (Tesseract, RapidOCR, EasyOCR via Docling + Google Vision)
on all 1,200 images. Supports resume via checkpoint.

Usage:
    python -m research.ocr_iqa_correlation.scripts.03_run_ocr [--engines tesseract,rapidocr]
"""

from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OCR engines on all images")
    parser.add_argument(
        "--engines",
        type=str,
        default=None,
        help="Comma-separated list of engines to run (default: all)",
    )
    parser.add_argument(
        "--skip-gcloud",
        action="store_true",
        help="Skip Google Cloud Vision (requires credentials)",
    )
    return parser.parse_args()


def main() -> None:
    """Run OCR on all images."""
    args = parse_args()

    from research.ocr_iqa_correlation.config import DATA_DIR, DEFAULT_CONFIG
    from research.ocr_iqa_correlation.ocr.runner import run_all_engines

    # Load distortion metadata to get image list
    distortion_meta = DATA_DIR / "distortion_metadata.jsonl"
    if not distortion_meta.exists():
        logger.error("Distortion metadata not found: %s", distortion_meta)
        logger.error("Run step 02 first.")
        return

    image_records = []
    with open(distortion_meta) as f:
        for line in f:
            image_records.append(json.loads(line))

    logger.info("Loaded %d image records", len(image_records))

    # Determine which engines to run
    engine_names = DEFAULT_CONFIG.ocr_engines
    if args.engines:
        engine_names = [e.strip() for e in args.engines.split(",")]
    if args.skip_gcloud and "gcloud_vision" in engine_names:
        engine_names.remove("gcloud_vision")

    # Create engine instances
    engines = []
    for name in engine_names:
        if name in ("tesseract", "rapidocr", "easyocr"):
            from research.ocr_iqa_correlation.ocr.docling_engines import (
                DoclingOCREngine,
            )
            engines.append(DoclingOCREngine(name))
        elif name == "gcloud_vision":
            from research.ocr_iqa_correlation.ocr.gcloud_vision import (
                GoogleVisionOCREngine,
            )
            engines.append(GoogleVisionOCREngine())
        else:
            logger.warning("Unknown engine: %s, skipping", name)

    logger.info("Running %d engines: %s", len(engines), [e.name for e in engines])

    # Run OCR
    result_paths = run_all_engines(engines, image_records)

    logger.info("=" * 60)
    logger.info("OCR complete:")
    for engine_name, path in result_paths.items():
        logger.info("  %s: %s", engine_name, path)


if __name__ == "__main__":
    main()
