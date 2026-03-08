"""Stratified sampling of base images from FUNSD and FUNSD+ datasets.

Selects 200 base images: ~50 from FUNSD (over-sampled for balance)
and ~150 from FUNSD+. Filters for minimum text length and uses a
fixed random seed for reproducibility.

After sampling, copies images locally to data/base_images/ so the
pipeline no longer depends on the external E: drive mount.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

from research.ocr_iqa_correlation.config import (
    BASE_IMAGES_DIR,
    DEFAULT_CONFIG,
    GT_TEXT_DIR,
    SAMPLE_MANIFEST,
    ExperimentConfig,
)
from research.ocr_iqa_correlation.gt_extraction.funsd_parser import extract_funsd_gt
from research.ocr_iqa_correlation.gt_extraction.funsd_plus_parser import (
    extract_funsd_plus_gt,
)

logger = logging.getLogger(__name__)


def _copy_images_locally(
    samples: list[dict[str, str]],
    base_images_dir: Path = BASE_IMAGES_DIR,
) -> list[dict[str, str]]:
    """Copy sampled images to a local directory and update paths.

    Args:
        samples: Sample records with image_path pointing to source.
        base_images_dir: Local directory to copy images into.

    Returns:
        Updated samples with image_path pointing to local copies.
    """
    base_images_dir.mkdir(parents=True, exist_ok=True)
    updated = []

    for sample in samples:
        src = Path(sample["image_path"])
        suffix = src.suffix
        local_path = base_images_dir / f"{sample['image_id']}{suffix}"

        if not local_path.exists():
            shutil.copy2(src, local_path)
            logger.debug("Copied %s → %s", src.name, local_path)

        updated_sample = dict(sample)
        updated_sample["source_image_path"] = sample["image_path"]
        updated_sample["image_path"] = str(local_path)
        updated.append(updated_sample)

    logger.info(
        "Copied %d images to %s (%.1f MB)",
        len(updated),
        base_images_dir,
        sum(Path(s["image_path"]).stat().st_size for s in updated) / 1e6,
    )
    return updated


def _save_gt_text_files(
    samples: list[dict[str, str]],
    gt_text_dir: Path = GT_TEXT_DIR,
) -> None:
    """Save ground truth text as individual .txt files.

    Args:
        samples: Sample records with text field.
        gt_text_dir: Directory to write GT text files.
    """
    gt_text_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        gt_path = gt_text_dir / f"{sample['image_id']}.txt"
        gt_path.write_text(sample["text"], encoding="utf-8")

    logger.info("Saved %d GT text files to %s", len(samples), gt_text_dir)


def sample_base_images(
    config: ExperimentConfig = DEFAULT_CONFIG,
    output_path: Path = SAMPLE_MANIFEST,
    copy_locally: bool = True,
) -> list[dict[str, str]]:
    """Sample base images from FUNSD and FUNSD+ with stratification.

    Extracts GT text, samples images, copies them locally, and saves
    both the manifest and per-image GT text files.

    Args:
        config: Experiment configuration with sample counts and seed.
        output_path: Path to write the sample manifest JSON.
        copy_locally: If True, copy images to data/base_images/ and
            update paths in the manifest to point locally.

    Returns:
        List of sample records with image_id, text, image_path, source_dataset.
    """
    rng = random.Random(config.random_seed)

    # Extract GT text from both datasets
    logger.info("Extracting FUNSD ground truth text...")
    funsd_gt = extract_funsd_gt()

    logger.info("Extracting FUNSD+ ground truth text...")
    funsd_plus_gt = extract_funsd_plus_gt()

    # Filter by minimum character count
    funsd_candidates = {
        k: v
        for k, v in funsd_gt.items()
        if len(v["text"]) >= config.min_gt_chars
    }
    funsd_plus_candidates = {
        k: v
        for k, v in funsd_plus_gt.items()
        if len(v["text"]) >= config.min_gt_chars
    }

    logger.info(
        "Candidates after filtering (min %d chars): FUNSD=%d, FUNSD+=%d",
        config.min_gt_chars,
        len(funsd_candidates),
        len(funsd_plus_candidates),
    )

    # Sample from each dataset
    funsd_keys = sorted(funsd_candidates.keys())
    funsd_plus_keys = sorted(funsd_plus_candidates.keys())

    funsd_n = min(config.funsd_sample_count, len(funsd_keys))
    funsd_plus_n = min(config.funsd_plus_sample_count, len(funsd_plus_keys))

    # If one dataset has fewer samples, redistribute to the other
    total_needed = config.total_base_images
    if funsd_n < config.funsd_sample_count:
        funsd_plus_n = min(total_needed - funsd_n, len(funsd_plus_keys))
    elif funsd_plus_n < config.funsd_plus_sample_count:
        funsd_n = min(total_needed - funsd_plus_n, len(funsd_keys))

    sampled_funsd = rng.sample(funsd_keys, funsd_n)
    sampled_funsd_plus = rng.sample(funsd_plus_keys, funsd_plus_n)

    # Build manifest
    samples: list[dict[str, str]] = []
    for key in sorted(sampled_funsd):
        entry = funsd_candidates[key]
        samples.append({
            "image_id": key,
            "text": entry["text"],
            "image_path": entry["image_path"],
            "source_dataset": entry["source_dataset"],
            "original_id": entry["original_id"],
            "gt_chars": str(len(entry["text"])),
        })

    for key in sorted(sampled_funsd_plus):
        entry = funsd_plus_candidates[key]
        samples.append({
            "image_id": key,
            "text": entry["text"],
            "image_path": entry["image_path"],
            "source_dataset": entry["source_dataset"],
            "original_id": entry["original_id"],
            "gt_chars": str(len(entry["text"])),
        })

    logger.info(
        "Sampled %d images: %d FUNSD + %d FUNSD+",
        len(samples),
        funsd_n,
        funsd_plus_n,
    )

    # Copy images locally so pipeline doesn't depend on E: drive
    if copy_locally:
        samples = _copy_images_locally(samples)

    # Save GT text files
    _save_gt_text_files(samples)

    # Write manifest (text field excluded to keep it compact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for s in samples:
        record = {k: v for k, v in s.items() if k != "text"}
        manifest_records.append(record)

    with open(output_path, "w") as f:
        json.dump(manifest_records, f, indent=2)

    logger.info("Sample manifest written to %s", output_path)
    return samples
