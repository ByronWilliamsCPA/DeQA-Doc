#!/usr/bin/env python3
"""Step 4: Run DeQA scoring on all images.

Must be run in the DeQA-Score venv due to torch/transformers version conflicts:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \\
        -m research.ocr_iqa_correlation.scripts.04_run_deqa

Usage:
    python -m research.ocr_iqa_correlation.scripts.04_run_deqa [--model-path PATH]
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
    parser = argparse.ArgumentParser(description="Run DeQA scoring")
    parser.add_argument(
        "--model-path",
        type=str,
        default="zhiyuanyou/DeQA-Score-Mix3",
        help="HuggingFace model path or local checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    return parser.parse_args()


def main() -> None:
    """Run DeQA scoring on all images."""
    args = parse_args()

    from research.ocr_iqa_correlation.config import DATA_DIR
    from research.ocr_iqa_correlation.deqa.scorer_wrapper import score_images

    # Load distortion metadata
    distortion_meta = DATA_DIR / "distortion_metadata.jsonl"
    if not distortion_meta.exists():
        logger.error("Distortion metadata not found: %s", distortion_meta)
        logger.error("Run step 02 first.")
        return

    image_records = []
    with open(distortion_meta) as f:
        for line in f:
            image_records.append(json.loads(line))

    logger.info("Loaded %d image records for scoring", len(image_records))

    results_path = score_images(
        image_records=image_records,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )

    logger.info("=" * 60)
    logger.info("DeQA scoring complete: %s", results_path)


if __name__ == "__main__":
    main()
