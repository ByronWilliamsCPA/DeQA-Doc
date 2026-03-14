#!/usr/bin/env python3
"""Step 2: Apply document-realistic distortions to sampled images.

Generates 5 distortion tiers (PRISTINE, HIGH, MEDIUM, LOW, DEGRADED)
plus ORIGINAL copies for all 200 base images = 1,200 total images.

Requires augraphy and albumentations. Install with:
    uv add augraphy albumentations

Usage:
    python -m research.ocr_iqa_correlation.scripts.02_apply_distortions
"""

from __future__ import annotations

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Apply distortions to all sampled base images."""
    from research.ocr_iqa_correlation.config import (
        DATA_DIR,
        DEFAULT_CONFIG,
        SAMPLE_MANIFEST,
    )
    from research.ocr_iqa_correlation.distortion.apply_distortions import (
        apply_all_distortions,
    )

    # Load sample manifest
    if not SAMPLE_MANIFEST.exists():
        logger.error("Sample manifest not found: %s", SAMPLE_MANIFEST)
        logger.error("Run step 01 first.")
        return

    with open(SAMPLE_MANIFEST) as f:
        samples = json.load(f)

    logger.info("Loaded %d samples from manifest", len(samples))

    # Apply distortions
    distortion_records = apply_all_distortions(
        sample_manifest=samples,
        base_seed=DEFAULT_CONFIG.base_distortion_seed,
        augraphy_probability=DEFAULT_CONFIG.augraphy_probability,
        albumentations_probability=DEFAULT_CONFIG.albumentations_probability,
    )

    # Save distortion metadata
    distortion_meta_path = DATA_DIR / "distortion_metadata.jsonl"
    with open(distortion_meta_path, "w") as f:
        for record in distortion_records:
            f.write(json.dumps(record) + "\n")

    # Summary
    from collections import Counter

    tier_counts = Counter(r["tier"] for r in distortion_records)
    logger.info("=" * 60)
    logger.info("Distortion complete:")
    logger.info("  Total images: %d", len(distortion_records))
    for tier, count in sorted(tier_counts.items()):
        logger.info("  %s: %d", tier, count)
    logger.info("  Metadata: %s", distortion_meta_path)


if __name__ == "__main__":
    main()
