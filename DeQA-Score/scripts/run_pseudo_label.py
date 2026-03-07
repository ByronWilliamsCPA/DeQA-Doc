#!/usr/bin/env python3
"""CLI entrypoint for the pseudo-labeling pipeline.

Processes SigLIP2 predictions through OOD detection, cross-validation,
uncertainty fusion, and optional VLM veto to produce confidence-weighted
pseudo-labels in DeQA training JSON format.

Usage:
    python scripts/run_pseudo_label.py \
        --siglip2-results predictions/siglip2_iqa.json \
        --deqa-specialist predictions/diqa-5000_specialist_true_labels.jsonl \
        --deqa-ensemble predictions/diqa-5000_ensemble_labels.jsonl \
        --embeddings /path/to/embeddings.npy \
        --ood-params /path/to/ood_params_4400.npz \
        --output-dir Data-DeQA-Score/pseudo/ \
        --dimensions overall sharpness color
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_pseudo_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pseudo-labeling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--siglip2-results",
        type=str,
        required=True,
        help="JSON file with SigLIP2 predictions (list of dicts with image_id, *_mu, *_sigma_sq)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="NPY file with SigLIP2 embeddings, shape (N, 768)",
    )
    parser.add_argument(
        "--ood-params",
        type=str,
        required=True,
        help="NPZ file with OOD detector parameters",
    )
    parser.add_argument(
        "--deqa-specialist",
        type=str,
        default=None,
        help="JSONL file with DeQA dimension-specific predictions",
    )
    parser.add_argument(
        "--deqa-ensemble",
        type=str,
        default=None,
        help="JSONL file with DeQA-Score-Mix3 ensemble predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output training JSON files",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["overall", "sharpness", "color"],
        help="Quality dimensions to process",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="",
        help="Prefix to prepend to image paths in output JSON",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Root directory for images (for VLM validation)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.3,
        help="Minimum confidence weight to include in output",
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=46.0,
        help="Mahalanobis distance OOD threshold",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Enable Tier-2 VLM validation via OpenRouter",
    )
    parser.add_argument(
        "--per-dimension",
        action="store_true",
        help="Output separate JSON files per dimension",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.uncertainty.cross_validator import CrossValidator
    from src.uncertainty.format_training_data import (
        generate_per_dimension_json,
        samples_to_training_json,
    )
    from src.uncertainty.fusion import UncertaintyFusion
    from src.uncertainty.ood_wrapper import OODDetectorWrapper
    from src.uncertainty.pseudo_label import PseudoLabelPipeline
    from src.uncertainty.vlm_validator import VLMValidator

    # Load SigLIP2 results
    logger.info("Loading SigLIP2 results from: %s", args.siglip2_results)
    with open(args.siglip2_results) as f:
        siglip2_outputs = json.load(f)
    logger.info("  Loaded %d image predictions", len(siglip2_outputs))

    # Load embeddings
    logger.info("Loading embeddings from: %s", args.embeddings)
    embeddings = np.load(args.embeddings)
    logger.info("  Shape: %s", embeddings.shape)

    if len(embeddings) != len(siglip2_outputs):
        logger.error(
            "Mismatch: %d embeddings vs %d predictions",
            len(embeddings),
            len(siglip2_outputs),
        )
        sys.exit(1)

    # Load OOD detector
    logger.info("Loading OOD detector from: %s", args.ood_params)
    ood_detector = OODDetectorWrapper.from_npz(
        args.ood_params, threshold=args.ood_threshold
    )

    # Load cross-validator
    if args.deqa_specialist or args.deqa_ensemble:
        logger.info("Loading DeQA cross-validator predictions")
        cross_validator = CrossValidator.from_jsonl(
            specialist_path=args.deqa_specialist,
            ensemble_path=args.deqa_ensemble,
        )
    else:
        logger.warning("No DeQA predictions provided. Cross-validation disabled.")
        cross_validator = CrossValidator({})

    # Create fusion engine
    fusion = UncertaintyFusion()

    # Create VLM validator (optional)
    vlm_validator = None
    if args.enable_vlm:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY not set. Cannot enable VLM validation.")
            sys.exit(1)
        vlm_validator = VLMValidator(api_key=api_key)
        logger.info("VLM validation enabled (Tier-2 veto)")

    # Create pipeline
    pipeline = PseudoLabelPipeline(
        ood_detector=ood_detector,
        cross_validator=cross_validator,
        fusion=fusion,
        vlm_validator=vlm_validator,
        image_root=args.image_root,
    )

    # Process batch
    logger.info(
        "Processing %d images across %s dimensions",
        len(siglip2_outputs),
        args.dimensions,
    )
    all_samples = pipeline.process_batch(
        siglip2_outputs=siglip2_outputs,
        embeddings=embeddings,
        dimensions=tuple(args.dimensions),
    )

    # Filter accepted samples
    accepted = pipeline.filter_accepted(all_samples, min_weight=args.min_weight)

    # Write output
    output_dir = Path(args.output_dir)
    if args.per_dimension:
        counts = generate_per_dimension_json(
            accepted,
            output_dir=output_dir,
            image_prefix=args.image_prefix,
            min_weight=args.min_weight,
            seed=args.seed,
        )
        for dim, count in counts.items():
            logger.info(
                "  %s: %d samples → %s", dim, count, output_dir / f"pseudo_{dim}.json"
            )
    else:
        output_path = output_dir / "pseudo_labels.json"
        count = samples_to_training_json(
            accepted,
            output_path=output_path,
            image_prefix=args.image_prefix,
            min_weight=args.min_weight,
            seed=args.seed,
        )
        logger.info("  Wrote %d samples → %s", count, output_path)

    # Summary
    logger.info("\n=== Pipeline Summary ===")
    logger.info("Total processed: %d", len(all_samples))
    logger.info("Accepted (weight >= %.2f): %d", args.min_weight, len(accepted))

    tier_counts: dict[str, int] = {}
    for s in all_samples:
        tier_counts[s.tier.value] = tier_counts.get(s.tier.value, 0) + 1
    for tier, count in sorted(tier_counts.items()):
        logger.info("  %s: %d (%.1f%%)", tier, count, count / len(all_samples) * 100)

    if vlm_validator:
        budget = vlm_validator.budget.summary()
        logger.info("\nVLM Budget: %s", budget)


if __name__ == "__main__":
    main()
