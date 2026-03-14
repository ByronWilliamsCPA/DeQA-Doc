"""Parse FUNSD+ Arrow dataset to extract ground truth text per image.

FUNSD+ uses HuggingFace Arrow format with features:
    - image: PIL Image
    - words: List[str] - word-level text (ground truth)
    - bboxes: List[List[float64]] - word bounding boxes
    - ner_tags: List[ClassLabel] - BIO NER tags

We join the `words` list with spaces to reconstruct full text.
Only training split is used (test split is benchmark reserved).
"""

from __future__ import annotations

import logging
from pathlib import Path

from research.ocr_iqa_correlation.config import (
    FUNSD_PLUS_IMAGES,
    FUNSD_PLUS_TRAIN_ARROW,
)

logger = logging.getLogger(__name__)


def extract_funsd_plus_gt(
    arrow_path: Path = FUNSD_PLUS_TRAIN_ARROW,
    images_dir: Path = FUNSD_PLUS_IMAGES,
) -> dict[str, dict[str, str]]:
    """Extract ground truth text for all FUNSD+ training images.

    Args:
        arrow_path: Path to the FUNSD+ training Arrow file.
        images_dir: Directory containing FUNSD+ image files.

    Returns:
        Dict mapping image_id to {"text": str, "image_path": str}.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        msg = "Install 'datasets' package: uv add datasets"
        raise ImportError(msg) from e

    if not arrow_path.exists():
        msg = f"FUNSD+ Arrow file not found: {arrow_path}"
        raise FileNotFoundError(msg)

    logger.info("Loading FUNSD+ training data from %s", arrow_path)
    dataset = Dataset.from_file(str(arrow_path))

    results: dict[str, dict[str, str]] = {}

    for idx in range(len(dataset)):
        row = dataset[idx]
        words = row.get("words", [])
        text = " ".join(words) if words else ""

        if not text.strip():
            logger.warning("Empty text for FUNSD+ index %d, skipping", idx)
            continue

        # FUNSD+ image filenames follow pattern: funsd_plus_train_NNNN.jpg
        image_filename = f"funsd_plus_train_{idx:04d}.jpg"
        image_path = images_dir / image_filename

        if not image_path.exists():
            logger.warning("Image not found: %s, skipping", image_path)
            continue

        image_id = f"funsd_plus_train_{idx:04d}"
        results[image_id] = {
            "text": text,
            "image_path": str(image_path),
            "source_dataset": "funsd_plus",
            "original_id": str(idx),
        }

    logger.info("Extracted GT text for %d FUNSD+ images", len(results))
    return results
