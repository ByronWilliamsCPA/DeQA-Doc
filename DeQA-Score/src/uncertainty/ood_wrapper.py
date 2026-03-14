"""Mahalanobis-distance OOD detector wrapper.

Wraps the pre-fitted OOD detector parameters (mean, precision matrix) from
the Tier 1 OOD detector. Scores SigLIP2 embeddings to flag out-of-distribution
documents relative to the DIQA-5000 training set.

Thresholds calibrated against ground truth labels (1,150 ID + 370 OOD synthetic):
    - 55.37: TPR=95%, FPR=14.6% (default — balanced operating point)
    - 61.62: TPR=80%, FPR=12.6% (conservative, used as hard_reject in fusion)

See research/ood_baselines/RESULTS.md for full evaluation methodology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Ground-truth-calibrated thresholds (eval_id_ood.npz, 1520 samples)
MAHALANOBIS_OOD_THRESHOLD = 55.37  # soft flag
MAHALANOBIS_HARD_REJECT = 61.62  # hard reject


@dataclass(frozen=True)
class OODResult:
    """Result of OOD scoring for a single embedding."""

    mahalanobis_distance: float
    is_ood: bool
    threshold: float


class OODDetectorWrapper:
    """Mahalanobis-distance OOD detector using pre-fitted parameters.

    Computes d = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ)) where μ and Σ⁻¹ are fitted on
    DIQA-5000 train+val embeddings with Ledoit-Wolf shrinkage.

    Thresholds calibrated against ground truth ID/OOD labels from synthetic
    evaluation set (1,150 ID + 370 OOD across 13 categories). See
    research/ood_baselines/RESULTS.md for methodology.

    Args:
        mean: Centroid of training embeddings, shape (768,).
        precision_matrix: Inverse covariance matrix, shape (768, 768).
        threshold: Mahalanobis distance above which an image is flagged OOD.
            Default 55.37 = ground-truth TPR@95% (14.6% FPR).
    """

    def __init__(
        self,
        mean: np.ndarray,
        precision_matrix: np.ndarray,
        threshold: float = MAHALANOBIS_OOD_THRESHOLD,
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.precision_matrix = np.asarray(precision_matrix, dtype=np.float64)
        self.threshold = threshold

        if self.mean.ndim != 1:
            msg = f"mean must be 1-D, got shape {self.mean.shape}"
            raise ValueError(msg)
        dim = self.mean.shape[0]
        if self.precision_matrix.shape != (dim, dim):
            msg = (
                f"precision_matrix shape {self.precision_matrix.shape} "
                f"does not match mean dimension {dim}"
            )
            raise ValueError(msg)

    @classmethod
    def from_npz(
        cls, path: str, threshold: float = MAHALANOBIS_OOD_THRESHOLD
    ) -> OODDetectorWrapper:
        """Load from .npz file produced by the OOD fitting script.

        Expected keys: 'mean', 'precision_matrix'. Optional: 'threshold',
        'calibration_distances'.
        """
        data = np.load(path)
        mean = data["mean"]
        precision_matrix = data["precision_matrix"]
        # Use stored threshold as fallback if not explicitly provided
        if "threshold" in data and threshold == MAHALANOBIS_OOD_THRESHOLD:
            threshold = float(data["threshold"])
        return cls(mean=mean, precision_matrix=precision_matrix, threshold=threshold)

    @classmethod
    def calibrate_from_ground_truth(
        cls,
        npz_path: str,
        eval_npz_path: str,
        target_tpr: float = 0.95,
    ) -> OODDetectorWrapper:
        """Load detector and calibrate threshold from ground truth eval data.

        Computes optimal Mahalanobis threshold at the given TPR operating point
        using labeled ID/OOD evaluation data.

        Args:
            npz_path: Path to detector .npz (mean, precision_matrix).
            eval_npz_path: Path to eval .npz with 'embeddings' and 'labels'
                (0=ID, 1=OOD).
            target_tpr: Desired true positive rate for OOD detection.

        Returns:
            OODDetectorWrapper with calibrated threshold.
        """
        data = np.load(npz_path)
        mean = data["mean"]
        precision_matrix = data["precision_matrix"]

        eval_data = np.load(eval_npz_path, allow_pickle=True)
        eval_emb = np.asarray(eval_data["embeddings"], dtype=np.float64)
        labels = eval_data["labels"]

        # Compute Mahalanobis distances for all eval samples
        diffs = eval_emb - mean[np.newaxis, :]
        transformed = diffs @ precision_matrix
        distances = np.sqrt(np.sum(transformed * diffs, axis=1))

        # Find threshold at target TPR
        ood_distances = distances[labels == 1]
        threshold = float(np.percentile(ood_distances, (1 - target_tpr) * 100))

        # Report calibration stats
        id_distances = distances[labels == 0]
        fpr = float(np.mean(id_distances > threshold))
        tpr = float(np.mean(ood_distances > threshold))
        logger.info(
            "Calibrated Mahalanobis threshold=%.2f at target_tpr=%.2f "
            "(actual TPR=%.4f, FPR=%.4f, n_id=%d, n_ood=%d)",
            threshold,
            target_tpr,
            tpr,
            fpr,
            len(id_distances),
            len(ood_distances),
        )

        return cls(mean=mean, precision_matrix=precision_matrix, threshold=threshold)

    def mahalanobis_distance(self, embedding: np.ndarray) -> float:
        """Compute Mahalanobis distance for a single embedding.

        Args:
            embedding: Shape (D,) where D matches the fitted dimension.

        Returns:
            Scalar Mahalanobis distance.
        """
        diff = np.asarray(embedding, dtype=np.float64) - self.mean
        return float(np.sqrt(diff @ self.precision_matrix @ diff))

    def score(self, embedding: np.ndarray) -> OODResult:
        """Score a single embedding and return OOD decision.

        Args:
            embedding: Shape (D,) SigLIP2 penultimate-layer embedding.
        """
        dist = self.mahalanobis_distance(embedding)
        return OODResult(
            mahalanobis_distance=dist,
            is_ood=dist > self.threshold,
            threshold=self.threshold,
        )

    def score_batch(self, embeddings: np.ndarray) -> list[OODResult]:
        """Score a batch of embeddings.

        Args:
            embeddings: Shape (N, D) array of embeddings.

        Returns:
            List of N OODResult instances.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        diffs = embeddings - self.mean[np.newaxis, :]
        # Vectorized: (N, D) @ (D, D) -> (N, D), then element-wise with diffs
        transformed = diffs @ self.precision_matrix
        distances = np.sqrt(np.sum(transformed * diffs, axis=1))
        return [
            OODResult(
                mahalanobis_distance=float(d),
                is_ood=d > self.threshold,
                threshold=self.threshold,
            )
            for d in distances
        ]
