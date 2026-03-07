"""Mahalanobis-distance OOD detector wrapper.

Wraps the pre-fitted OOD detector parameters (mean, precision matrix) from
the Tier 1 OOD detector. Scores SigLIP2 embeddings to flag out-of-distribution
documents relative to the DIQA-5000 training set.

See results/tier1_ood_detector/README.md for full methodology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    Args:
        mean: Centroid of training embeddings, shape (768,).
        precision_matrix: Inverse covariance matrix, shape (768, 768).
        threshold: Mahalanobis distance above which an image is flagged OOD.
            Default 46.0 = test p95 (5% FPR, 99.5% OOD TPR).
    """

    def __init__(
        self,
        mean: np.ndarray,
        precision_matrix: np.ndarray,
        threshold: float = 46.0,
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
    def from_npz(cls, path: str, threshold: float = 46.0) -> OODDetectorWrapper:
        """Load from .npz file produced by the OOD fitting script.

        Expected keys: 'mean', 'precision_matrix'. Optional: 'threshold',
        'calibration_distances'.
        """
        data = np.load(path)
        mean = data["mean"]
        precision_matrix = data["precision_matrix"]
        # Use stored threshold as fallback if not explicitly provided
        if "threshold" in data and threshold == 46.0:
            threshold = float(data["threshold"])
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
