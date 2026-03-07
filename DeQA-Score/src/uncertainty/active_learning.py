"""Active learning sample selection using BALD scores.

Selects the most informative unlabeled samples for human annotation
by measuring epistemic uncertainty via Bayesian Active Learning by
Disagreement (BALD) between SigLIP2 and DeQA model predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .discrete_metrics import bald_score, discrete_entropy
from .pseudo_label import PseudoLabelSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActiveLearningSample:
    """A sample scored for active learning selection."""

    image_id: str
    dimension: str
    bald: float
    entropy: float
    mos: float


@dataclass
class ConvergenceMetrics:
    """Tracks convergence across active learning iterations."""

    iteration: int
    srcc_delta: float
    plcc_delta: float
    mean_bald: float
    samples_above_threshold: int

    @property
    def is_converged(self) -> bool:
        """Check if metrics indicate convergence (SRCC delta < 0.01)."""
        return abs(self.srcc_delta) < 0.01


class ActiveLearningSelector:
    """Selects informative samples for human annotation using BALD.

    BALD score = H[avg distribution] - avg[H[each distribution]]
    High BALD = models disagree = high epistemic uncertainty = most informative.

    Args:
        sacred_test_ids: Set of image IDs that must NEVER appear in training.
    """

    def __init__(self, sacred_test_ids: set[str] | None = None) -> None:
        self.sacred_test_ids: set[str] = sacred_test_ids or set()

    @classmethod
    def from_sacred_ids_file(cls, path: str | Path) -> ActiveLearningSelector:
        """Load sacred test IDs from JSON file.

        Args:
            path: Path to JSON file containing list of sacred test image IDs.
        """
        with open(path) as f:
            ids = json.load(f)
        if not isinstance(ids, list):
            msg = f"Expected list of IDs, got {type(ids)}"
            raise ValueError(msg)
        return cls(sacred_test_ids=set(ids))

    def score_samples(
        self,
        pseudo_samples: list[PseudoLabelSample],
        deqa_probs_lookup: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> list[ActiveLearningSample]:
        """Compute BALD scores for pseudo-labeled samples.

        Uses SigLIP2's level_probs and (optionally) DeQA's level_probs as a
        2-member ensemble for BALD computation.

        Args:
            pseudo_samples: Samples from the pseudo-label pipeline.
            deqa_probs_lookup: Optional dict[dimension][image_id] → level_probs
                from DeQA cross-validators. If provided, used as second
                ensemble member for BALD.

        Returns:
            List of ActiveLearningSample sorted by BALD score (descending).
        """
        scored: list[ActiveLearningSample] = []

        for sample in pseudo_samples:
            if sample.image_id in self.sacred_test_ids:
                continue

            distributions = [sample.level_probs]

            if deqa_probs_lookup is not None:
                dim_lookup = deqa_probs_lookup.get(sample.dimension, {})
                deqa_probs = dim_lookup.get(sample.image_id)
                if deqa_probs is not None:
                    distributions.append(deqa_probs)

            bald = bald_score(distributions)
            entropy = discrete_entropy(sample.level_probs)

            scored.append(
                ActiveLearningSample(
                    image_id=sample.image_id,
                    dimension=sample.dimension,
                    bald=bald,
                    entropy=entropy,
                    mos=sample.mos,
                )
            )

        scored.sort(key=lambda s: s.bald, reverse=True)
        return scored

    def select_batch(
        self,
        scored_samples: list[ActiveLearningSample],
        already_labeled: set[str] | None = None,
        k: int = 1000,
    ) -> list[ActiveLearningSample]:
        """Select top-k most informative samples for annotation.

        Args:
            scored_samples: BALD-scored samples (should be pre-sorted).
            already_labeled: Image IDs already annotated (skip these).
            k: Number of samples to select.

        Returns:
            Top-k samples not in already_labeled or sacred_test_ids.
        """
        already_labeled = already_labeled or set()
        selected: list[ActiveLearningSample] = []

        for sample in scored_samples:
            if len(selected) >= k:
                break
            if sample.image_id in already_labeled:
                continue
            if sample.image_id in self.sacred_test_ids:
                continue
            selected.append(sample)

        logger.info(
            "Selected %d/%d samples for annotation (requested %d)",
            len(selected),
            len(scored_samples),
            k,
        )
        return selected

    def generate_annotation_queue(
        self,
        selected: list[ActiveLearningSample],
        output_path: str | Path,
    ) -> int:
        """Write annotation queue to JSON file.

        Args:
            selected: Samples selected for annotation.
            output_path: Path for output JSON file.

        Returns:
            Number of samples written.
        """
        records = [
            {
                "image_id": s.image_id,
                "dimension": s.dimension,
                "bald_score": round(s.bald, 6),
                "entropy": round(s.entropy, 6),
                "predicted_mos": round(s.mos, 4),
            }
            for s in selected
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(records, f, indent=2)

        logger.info(
            "Wrote %d samples to annotation queue: %s", len(records), output_path
        )
        return len(records)

    @staticmethod
    def check_convergence(
        iteration_metrics: list[dict[str, float]],
        srcc_threshold: float = 0.01,
    ) -> ConvergenceMetrics:
        """Check if active learning has converged.

        Convergence = SRCC improvement < threshold between last two iterations.

        Args:
            iteration_metrics: List of dicts with keys 'iteration', 'srcc', 'plcc',
                'mean_bald', 'samples_above_threshold'.
            srcc_threshold: SRCC delta below which we consider converged.

        Returns:
            ConvergenceMetrics for the latest iteration.
        """
        if len(iteration_metrics) < 2:
            latest = iteration_metrics[-1] if iteration_metrics else {}
            return ConvergenceMetrics(
                iteration=int(latest.get("iteration", 0)),
                srcc_delta=1.0,
                plcc_delta=1.0,
                mean_bald=latest.get("mean_bald", 0.0),
                samples_above_threshold=int(latest.get("samples_above_threshold", 0)),
            )

        prev = iteration_metrics[-2]
        curr = iteration_metrics[-1]

        return ConvergenceMetrics(
            iteration=int(curr["iteration"]),
            srcc_delta=curr["srcc"] - prev["srcc"],
            plcc_delta=curr["plcc"] - prev["plcc"],
            mean_bald=curr.get("mean_bald", 0.0),
            samples_above_threshold=int(curr.get("samples_above_threshold", 0)),
        )
