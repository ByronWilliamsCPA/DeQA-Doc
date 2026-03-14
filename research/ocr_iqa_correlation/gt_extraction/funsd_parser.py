"""Parse FUNSD JSON annotations to extract ground truth text per image.

FUNSD annotation format:
    {
        "form": [
            {"id": 0, "text": "R&D", "label": "other", "words": [...], ...},
            {"id": 1, "text": "some text", "label": "question", ...},
            ...
        ]
    }

We concatenate entity-level `text` fields sorted by `id` to approximate
reading order. This gives better text than word-level reconstruction since
entity text preserves intra-entity spacing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research.ocr_iqa_correlation.config import (
    FUNSD_TRAIN_ANNOTATIONS,
    FUNSD_TRAIN_IMAGES,
)

logger = logging.getLogger(__name__)


def parse_funsd_annotation(annotation_path: Path) -> str:
    """Extract full text from a single FUNSD annotation file.

    Args:
        annotation_path: Path to a FUNSD annotation JSON file.

    Returns:
        Concatenated text from all form entities, sorted by entity id.
    """
    with open(annotation_path) as f:
        data = json.load(f)

    entities = data.get("form", [])
    # Sort by id to approximate reading order
    sorted_entities = sorted(entities, key=lambda e: e.get("id", 0))
    texts = [e["text"] for e in sorted_entities if e.get("text", "").strip()]
    return " ".join(texts)


def extract_funsd_gt(
    annotations_dir: Path = FUNSD_TRAIN_ANNOTATIONS,
    images_dir: Path = FUNSD_TRAIN_IMAGES,
) -> dict[str, dict[str, str]]:
    """Extract ground truth text for all FUNSD training images.

    Args:
        annotations_dir: Directory containing FUNSD annotation JSON files.
        images_dir: Directory containing FUNSD image files.

    Returns:
        Dict mapping image_id to {"text": str, "image_path": str}.
        Only includes images that have both annotation and image files.
    """
    results: dict[str, dict[str, str]] = {}

    if not annotations_dir.exists():
        msg = f"FUNSD annotations directory not found: {annotations_dir}"
        raise FileNotFoundError(msg)

    annotation_files = sorted(annotations_dir.glob("*.json"))
    logger.info("Found %d FUNSD annotation files in %s", len(annotation_files), annotations_dir)

    for ann_path in annotation_files:
        image_id = ann_path.stem
        image_path = images_dir / f"{image_id}.png"

        if not image_path.exists():
            logger.warning("Image not found for %s, skipping", image_id)
            continue

        text = parse_funsd_annotation(ann_path)
        if not text.strip():
            logger.warning("Empty text for %s, skipping", image_id)
            continue

        results[f"funsd_{image_id}"] = {
            "text": text,
            "image_path": str(image_path),
            "source_dataset": "funsd",
            "original_id": image_id,
        }

    logger.info("Extracted GT text for %d FUNSD images", len(results))
    return results
