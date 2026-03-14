#!/usr/bin/env python3
"""Step 1: Extract ground truth text, sample 200 base images, copy locally.

Parses FUNSD and FUNSD+ annotations, filters by minimum text length,
creates a stratified sample of 200 images, copies them to data/base_images/,
and saves per-image GT text files.

After this step, the pipeline no longer depends on the E: drive mount.

Usage:
    python -m research.ocr_iqa_correlation.scripts.01_extract_gt_and_sample
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Extract GT text, sample images, and copy locally."""
    from research.ocr_iqa_correlation.config import (
        BASE_IMAGES_DIR,
        DEFAULT_CONFIG,
        GT_TEXT_DIR,
        SAMPLE_MANIFEST,
    )
    from research.ocr_iqa_correlation.gt_extraction.sampler import sample_base_images

    logger.info("Starting GT extraction and sampling...")
    logger.info(
        "Config: %d total images (%d FUNSD + %d FUNSD+), seed=%d",
        DEFAULT_CONFIG.total_base_images,
        DEFAULT_CONFIG.funsd_sample_count,
        DEFAULT_CONFIG.funsd_plus_sample_count,
        DEFAULT_CONFIG.random_seed,
    )

    samples = sample_base_images(
        config=DEFAULT_CONFIG,
        output_path=SAMPLE_MANIFEST,
        copy_locally=True,
    )

    # Summary
    funsd_count = sum(1 for s in samples if s["source_dataset"] == "funsd")
    funsd_plus_count = sum(1 for s in samples if s["source_dataset"] == "funsd_plus")
    char_counts = [len(s.get("text", "")) for s in samples]

    logger.info("=" * 60)
    logger.info("Sampling complete:")
    logger.info("  Total: %d images", len(samples))
    logger.info("  FUNSD: %d", funsd_count)
    logger.info("  FUNSD+: %d", funsd_plus_count)
    if char_counts:
        logger.info(
            "  GT chars: min=%d, max=%d, mean=%.0f",
            min(char_counts),
            max(char_counts),
            sum(char_counts) / len(char_counts),
        )
    logger.info("  Base images: %s", BASE_IMAGES_DIR)
    logger.info("  GT text: %s", GT_TEXT_DIR)
    logger.info("  Manifest: %s", SAMPLE_MANIFEST)


if __name__ == "__main__":
    main()
