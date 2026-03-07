#!/usr/bin/env python3
"""CLI for active learning iteration cycle.

Selects the most informative unlabeled samples for human annotation
using BALD scores, generates annotation queues, and tracks convergence
across iterations.

Usage:
    python scripts/run_active_learning.py \
        --pseudo-label-dir Data-DeQA-Score/pseudo/ \
        --deqa-specialist predictions/diqa-5000_specialist_true_labels.jsonl \
        --sacred-test-ids artifacts/sacred_test_ids.json \
        --output-queue artifacts/annotation_queue.json \
        --k 1000 \
        [--already-labeled artifacts/labeled_ids.json] \
        [--iteration-log artifacts/iteration_metrics.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("run_active_learning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Active learning sample selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pseudo-label-dir",
        type=str,
        required=True,
        help="Directory containing pseudo-label JSON files",
    )
    parser.add_argument(
        "--deqa-specialist",
        type=str,
        default=None,
        help="JSONL with DeQA predictions (for BALD ensemble)",
    )
    parser.add_argument(
        "--sacred-test-ids",
        type=str,
        default=None,
        help="JSON file with sacred test IDs (never include in training)",
    )
    parser.add_argument(
        "--already-labeled",
        type=str,
        default=None,
        help="JSON file with already-labeled image IDs to skip",
    )
    parser.add_argument(
        "--output-queue",
        type=str,
        required=True,
        help="Output path for annotation queue JSON",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of samples to select for annotation",
    )
    parser.add_argument(
        "--iteration-log",
        type=str,
        default=None,
        help="Path to iteration metrics JSON for convergence tracking",
    )
    return parser.parse_args()


def load_pseudo_samples(pseudo_dir: Path) -> list:
    """Load pseudo-label samples from JSON files."""
    from src.uncertainty.fusion import (
        AcceptanceDecision,
        AcceptanceTier,
        UncertaintySignals,
    )
    from src.uncertainty.pseudo_label import PseudoLabelSample

    samples = []
    for json_path in sorted(pseudo_dir.glob("pseudo_*.json")):
        logger.info("Loading: %s", json_path)
        with open(json_path) as f:
            records = json.load(f)

        for rec in records:
            level_probs = np.array(rec["level_probs"], dtype=np.float64)
            # Create a minimal AcceptanceDecision for the sample
            tier = AcceptanceTier(rec.get("source_tier", "auto_accept"))
            dummy_signals = UncertaintySignals(
                mahalanobis_distance=0.0,
                cross_model_jsd=0.0,
                siglip2_sigma_sq=0.0,
                siglip2_entropy=0.0,
            )
            decision = AcceptanceDecision(
                image_id=rec.get("id", ""),
                dimension="overall",
                tier=tier,
                confidence_weight=rec.get("confidence_weight", 1.0),
                signals=dummy_signals,
                reason="loaded from file",
            )

            # Extract dimension from ID (pseudo_{dim}_{image_id})
            parts = rec.get("id", "").split("_", 2)
            dimension = parts[1] if len(parts) >= 3 else "overall"

            samples.append(
                PseudoLabelSample(
                    image_id=rec.get("image", rec.get("id", "")),
                    dimension=dimension,
                    level_probs=level_probs,
                    mos=rec.get("gt_score", 3.0),
                    std=rec.get("std", 0.0),
                    confidence_weight=rec.get("confidence_weight", 1.0),
                    tier=tier,
                    decision=decision,
                    vlm_vetoed=False,
                )
            )

    logger.info("Loaded %d pseudo-label samples total", len(samples))
    return samples


def load_deqa_probs(
    specialist_path: str | None,
) -> dict[str, dict[str, np.ndarray]] | None:
    """Load DeQA predictions for BALD ensemble."""
    if not specialist_path:
        return None

    from src.uncertainty.cross_validator import (
        _extract_level_probs_from_deqa,
        _load_deqa_jsonl,
    )

    logger.info("Loading DeQA predictions from: %s", specialist_path)
    records = _load_deqa_jsonl(specialist_path)

    lookup: dict[str, dict[str, np.ndarray]] = {}
    for rec in records:
        image_id = rec.get("image", "")
        probs = _extract_level_probs_from_deqa(rec)
        dimension = rec.get("dimension", "overall")
        lookup.setdefault(dimension, {})[image_id] = probs

    return lookup


def main() -> None:
    args = parse_args()

    from src.uncertainty.active_learning import ActiveLearningSelector

    # Load sacred test IDs
    if args.sacred_test_ids:
        selector = ActiveLearningSelector.from_sacred_ids_file(args.sacred_test_ids)
        logger.info("Loaded %d sacred test IDs", len(selector.sacred_test_ids))
    else:
        selector = ActiveLearningSelector()
        logger.warning("No sacred test IDs provided. All images eligible.")

    # Load pseudo-label samples
    pseudo_dir = Path(args.pseudo_label_dir)
    if not pseudo_dir.exists():
        logger.error("Pseudo-label directory not found: %s", pseudo_dir)
        sys.exit(1)

    pseudo_samples = load_pseudo_samples(pseudo_dir)
    if not pseudo_samples:
        logger.error("No pseudo-label samples found")
        sys.exit(1)

    # Load DeQA probs for BALD ensemble
    deqa_probs = load_deqa_probs(args.deqa_specialist)

    # Score samples
    logger.info("Computing BALD scores...")
    scored = selector.score_samples(pseudo_samples, deqa_probs_lookup=deqa_probs)
    logger.info(
        "Scored %d samples (top BALD=%.4f)",
        len(scored),
        scored[0].bald if scored else 0,
    )

    # Load already-labeled IDs
    already_labeled: set[str] = set()
    if args.already_labeled:
        with open(args.already_labeled) as f:
            already_labeled = set(json.load(f))
        logger.info("Loaded %d already-labeled IDs", len(already_labeled))

    # Select batch
    selected = selector.select_batch(scored, already_labeled=already_labeled, k=args.k)

    # Write annotation queue
    n_written = selector.generate_annotation_queue(selected, args.output_queue)
    logger.info("Wrote %d samples to: %s", n_written, args.output_queue)

    # BALD statistics
    if scored:
        bald_scores = [s.bald for s in scored]
        logger.info("\n=== BALD Score Distribution ===")
        logger.info("  mean  = %.4f", np.mean(bald_scores))
        logger.info("  p50   = %.4f", np.median(bald_scores))
        logger.info("  p90   = %.4f", np.percentile(bald_scores, 90))
        logger.info("  p99   = %.4f", np.percentile(bald_scores, 99))
        logger.info("  max   = %.4f", np.max(bald_scores))

    # Convergence check
    if args.iteration_log and Path(args.iteration_log).exists():
        with open(args.iteration_log) as f:
            iteration_metrics = json.load(f)
        convergence = selector.check_convergence(iteration_metrics)
        logger.info("\n=== Convergence Check ===")
        logger.info("  Iteration: %d", convergence.iteration)
        logger.info("  SRCC delta: %.4f", convergence.srcc_delta)
        logger.info("  Converged: %s", convergence.is_converged)


if __name__ == "__main__":
    main()
