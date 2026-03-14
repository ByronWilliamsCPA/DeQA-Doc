"""Apply document-realistic distortions using the hybrid augmentation pipeline.

Imports HybridAugmentationPipeline from the image_detection project via
sys.path injection. Generates 5 distortion tiers per base image with
deterministic seeds for reproducibility.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from research.ocr_iqa_correlation.config import (
    BASE_IMAGES_DIR,
    DISTORTED_DIR,
    DISTORTION_TIERS,
    IMAGE_DETECTION_SRC,
    TIER_TO_HYBRID_PROFILE,
    QualityTier,
)

logger = logging.getLogger(__name__)


def _ensure_hybrid_pipeline():
    """Import HybridAugmentationPipeline from image_detection project."""
    src_path = str(IMAGE_DETECTION_SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from image_preprocessing_detector.synthetic.augmentation_hybrid import (
        HybridAugmentationPipeline,
        HybridProfile,
    )

    return HybridAugmentationPipeline, HybridProfile


def apply_distortions_to_image(
    image_path: str,
    image_id: str,
    base_seed: int,
    image_idx: int,
    augraphy_probability: float = 0.7,
    albumentations_probability: float = 1.0,
) -> list[dict]:
    """Apply all 5 distortion tiers to a single base image.

    Args:
        image_path: Path to the source image.
        image_id: Unique identifier for this image.
        base_seed: Base random seed for reproducibility.
        image_idx: Index of this image in the sample set.
        augraphy_probability: Probability of applying Augraphy effects.
        albumentations_probability: Probability of applying Albumentations.

    Returns:
        List of metadata dicts for each distorted image (including ORIGINAL).
    """
    HybridAugmentationPipeline, HybridProfile = _ensure_hybrid_pipeline()

    image = Image.open(image_path).convert("RGB")
    results = []

    # Copy original
    original_dir = DISTORTED_DIR / QualityTier.ORIGINAL.value
    original_dir.mkdir(parents=True, exist_ok=True)
    original_output = original_dir / f"{image_id}.png"
    image.save(original_output, "PNG")

    results.append({
        "image_id": image_id,
        "tier": QualityTier.ORIGINAL.value,
        "image_path": str(original_output),
        "seed": None,
        "iqa_labels": {
            "blur": 0.0,
            "noise": 0.0,
            "compression": 0.0,
            "ink_degradation": 0.0,
            "paper_degradation": 0.0,
            "geometric_distortion": 0.0,
            "bleed_through": 0.0,
            "overall_quality": 1.0,
        },
    })

    # Apply each distortion tier
    for tier_idx, tier in enumerate(DISTORTION_TIERS):
        seed = base_seed + image_idx * 100 + tier_idx
        profile_name = TIER_TO_HYBRID_PROFILE[tier]
        profile = HybridProfile(profile_name)

        pipeline = HybridAugmentationPipeline(
            seed=seed,
            augraphy_probability=augraphy_probability,
            albumentations_probability=albumentations_probability,
        )

        distorted_image, iqa_labels = pipeline.apply(image, profile=profile)

        # Save distorted image
        tier_dir = DISTORTED_DIR / tier.value
        tier_dir.mkdir(parents=True, exist_ok=True)
        output_path = tier_dir / f"{image_id}.png"
        distorted_image.save(output_path, "PNG")

        results.append({
            "image_id": image_id,
            "tier": tier.value,
            "image_path": str(output_path),
            "seed": seed,
            "iqa_labels": asdict(iqa_labels),
        })

        logger.debug(
            "%s / %s: overall_quality=%.3f (seed=%d)",
            image_id,
            tier.value,
            iqa_labels.overall_quality,
            seed,
        )

    return results


def apply_all_distortions(
    sample_manifest: list[dict],
    base_seed: int = 10000,
    augraphy_probability: float = 0.7,
    albumentations_probability: float = 1.0,
) -> list[dict]:
    """Apply distortions to all sampled base images.

    Args:
        sample_manifest: List of sample records from sampler.
        base_seed: Base seed for reproducible distortions.
        augraphy_probability: Probability of Augraphy effects per image.
        albumentations_probability: Probability of Albumentations effects.

    Returns:
        List of all distortion metadata records.
    """
    all_results = []

    for idx, sample in enumerate(sample_manifest):
        image_id = sample["image_id"]
        image_path = sample["image_path"]

        logger.info(
            "Distorting image %d/%d: %s", idx + 1, len(sample_manifest), image_id
        )

        results = apply_distortions_to_image(
            image_path=image_path,
            image_id=image_id,
            base_seed=base_seed,
            image_idx=idx,
            augraphy_probability=augraphy_probability,
            albumentations_probability=albumentations_probability,
        )
        all_results.extend(results)

    logger.info(
        "Generated %d distorted images from %d base images",
        len(all_results),
        len(sample_manifest),
    )
    return all_results
