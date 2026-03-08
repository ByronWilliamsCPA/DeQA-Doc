"""DeQA model scoring wrapper.

Wraps DeQA-Score's Scorer class for batch inference. Must be run in the
DeQA-Score venv (torch 2.0.1, transformers 4.36.1) due to dependency conflicts.

Usage from DeQA-Score directory:
    PYTHONPATH=./ .venv/bin/python -m research.ocr_iqa_correlation.scripts.04_run_deqa
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from research.ocr_iqa_correlation.config import DEQA_RESULTS_DIR

logger = logging.getLogger(__name__)

# DeQA level ordering: [excellent, good, fair, poor, bad] = [5, 4, 3, 2, 1]
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]
LEVEL_SCORES = [5.0, 4.0, 3.0, 2.0, 1.0]


def score_images(
    image_records: list[dict],
    model_path: str = "zhiyuanyou/DeQA-Score-Mix3",
    batch_size: int = 8,
    output_dir: Path = DEQA_RESULTS_DIR,
) -> Path:
    """Score images using the DeQA model.

    Args:
        image_records: List of dicts with image_id, tier, image_path.
        model_path: HuggingFace model path or local checkpoint.
        batch_size: Number of images per batch.
        output_dir: Directory for output JSONL.

    Returns:
        Path to the output JSONL file.
    """
    import numpy as np
    from PIL import Image

    # Import scorer from DeQA-Score (requires PYTHONPATH=./)
    from src.evaluate.scorer import Scorer

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "deqa_scores.jsonl"

    # Load completed records for resume
    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                record = json.loads(line)
                completed.add(f"{record['image_id']}_{record['tier']}")

    remaining = [
        r
        for r in image_records
        if f"{r['image_id']}_{r['tier']}" not in completed
    ]

    logger.info(
        "DeQA scoring: %d total, %d completed, %d remaining",
        len(image_records),
        len(completed),
        len(remaining),
    )

    if not remaining:
        return results_path

    # Initialize scorer
    scorer = Scorer(model_path=model_path)

    with open(results_path, "a") as f:
        for start_idx in range(0, len(remaining), batch_size):
            batch = remaining[start_idx : start_idx + batch_size]
            images = [
                Image.open(r["image_path"]).convert("RGB") for r in batch
            ]

            # Scorer returns MOS scores
            mos_scores = scorer(images).tolist()

            for record, mos in zip(batch, mos_scores):
                output = {
                    "image_id": record["image_id"],
                    "tier": record["tier"],
                    "deqa_mos": round(mos, 4),
                }
                f.write(json.dumps(output) + "\n")
                f.flush()

            if (start_idx + batch_size) % (batch_size * 10) == 0:
                logger.info(
                    "DeQA scoring: %d/%d",
                    min(start_idx + batch_size, len(remaining)),
                    len(remaining),
                )

    logger.info("DeQA scoring complete. Results at %s", results_path)
    return results_path
