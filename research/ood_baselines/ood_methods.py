"""OOD detection scoring methods on SigLIP2 embeddings.

Implements four OOD scoring functions for baseline comparison against the
existing Mahalanobis detector. All methods operate on pre-extracted 768-dim
SigLIP2 embeddings and return scores where higher = more OOD.

Reference: Paper 4 peer review — "No baseline OOD method comparisons
(cosine, KNN, energy, GMM, one-class SVM) on same embeddings."
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity


def mahalanobis_scores(
    train_emb: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    """Mahalanobis distance with Ledoit-Wolf shrinkage.

    Recomputed from scratch for fair comparison (same fitting procedure
    as the production detector in ood_wrapper.py).

    Args:
        train_emb: Reference embeddings, shape (N_train, D).
        test_emb: Test embeddings, shape (N_test, D).

    Returns:
        Mahalanobis distances, shape (N_test,). Higher = more OOD.
    """
    train_f64 = np.asarray(train_emb, dtype=np.float64)
    test_f64 = np.asarray(test_emb, dtype=np.float64)

    lw = LedoitWolf()
    lw.fit(train_f64)
    mean = train_f64.mean(axis=0)
    precision = lw.precision_

    diffs = test_f64 - mean[np.newaxis, :]
    # (N, D) @ (D, D) -> (N, D), then row-wise dot with diffs
    transformed = diffs @ precision
    return np.sqrt(np.sum(transformed * diffs, axis=1))


def knn_scores(
    train_emb: np.ndarray, test_emb: np.ndarray, k: int = 10
) -> np.ndarray:
    """Mean Euclidean distance to k nearest training neighbors.

    Args:
        train_emb: Reference embeddings, shape (N_train, D).
        test_emb: Test embeddings, shape (N_test, D).
        k: Number of nearest neighbors. Clamped to N_train if too large.

    Returns:
        Mean k-NN distances, shape (N_test,). Higher = more OOD.
    """
    k = min(k, train_emb.shape[0])
    tree = KDTree(train_emb)
    distances, _ = tree.query(test_emb, k=k)
    if k == 1:
        return distances.ravel()
    return distances.mean(axis=1)


def cosine_scores(
    train_emb: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    """One minus maximum cosine similarity to any training sample.

    Args:
        train_emb: Reference embeddings, shape (N_train, D).
        test_emb: Test embeddings, shape (N_test, D).

    Returns:
        Cosine OOD scores, shape (N_test,). Higher = more OOD.
    """
    # (N_test, N_train) similarity matrix
    sim = cosine_similarity(test_emb, train_emb)
    return 1.0 - sim.max(axis=1)


def energy_scores(
    train_emb: np.ndarray, test_emb: np.ndarray
) -> np.ndarray:
    """Negative log-sum-exp of cosine similarities (energy-based).

    For each test sample, computes -log(sum(exp(cos_sim))) across all
    training samples. Lower energy (more negative) = more in-distribution,
    so we negate to get higher = more OOD.

    Args:
        train_emb: Reference embeddings, shape (N_train, D).
        test_emb: Test embeddings, shape (N_test, D).

    Returns:
        Energy OOD scores, shape (N_test,). Higher = more OOD.
    """
    sim = cosine_similarity(test_emb, train_emb)
    # logsumexp for numerical stability
    max_sim = sim.max(axis=1, keepdims=True)
    lse = max_sim.ravel() + np.log(np.sum(np.exp(sim - max_sim), axis=1))
    # Negate: higher = more OOD
    return -lse
