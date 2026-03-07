"""Tests for ood_wrapper.py."""

import numpy as np
import pytest

from src.uncertainty.ood_wrapper import OODDetectorWrapper


@pytest.fixture
def simple_detector():
    """Create a simple 3-D detector for testing."""
    mean = np.array([0.0, 0.0, 0.0])
    # Identity precision matrix → Mahalanobis = Euclidean
    precision = np.eye(3)
    return OODDetectorWrapper(mean=mean, precision_matrix=precision, threshold=5.0)


class TestOODDetectorWrapper:
    def test_at_centroid(self, simple_detector):
        """Point at centroid → distance = 0."""
        result = simple_detector.score(np.array([0.0, 0.0, 0.0]))
        assert result.mahalanobis_distance == pytest.approx(0.0, abs=1e-10)
        assert not result.is_ood

    def test_known_distance(self, simple_detector):
        """With identity precision, Mahalanobis = Euclidean."""
        result = simple_detector.score(np.array([3.0, 4.0, 0.0]))
        assert result.mahalanobis_distance == pytest.approx(5.0, abs=1e-10)

    def test_ood_flagging(self, simple_detector):
        """Point beyond threshold should be flagged OOD."""
        result = simple_detector.score(np.array([10.0, 0.0, 0.0]))
        assert result.is_ood
        assert result.mahalanobis_distance > 5.0

    def test_threshold_stored(self, simple_detector):
        result = simple_detector.score(np.array([0.0, 0.0, 0.0]))
        assert result.threshold == 5.0

    def test_batch_scoring(self, simple_detector):
        embeddings = np.array(
            [
                [0.0, 0.0, 0.0],  # at centroid
                [3.0, 4.0, 0.0],  # distance = 5
                [10.0, 0.0, 0.0],  # distance = 10
            ]
        )
        results = simple_detector.score_batch(embeddings)
        assert len(results) == 3
        assert results[0].mahalanobis_distance == pytest.approx(0.0, abs=1e-10)
        assert results[1].mahalanobis_distance == pytest.approx(5.0, abs=1e-10)
        assert results[2].mahalanobis_distance == pytest.approx(10.0, abs=1e-10)
        assert not results[0].is_ood
        assert results[2].is_ood

    def test_non_identity_precision(self):
        """Test with scaled precision matrix."""
        mean = np.array([0.0, 0.0])
        # Scale dimension 0 by 4 (precision = 4 in that direction)
        precision = np.array([[4.0, 0.0], [0.0, 1.0]])
        detector = OODDetectorWrapper(
            mean=mean, precision_matrix=precision, threshold=10.0
        )
        # Point [1, 0]: d = sqrt(1*4*1) = 2
        result = detector.score(np.array([1.0, 0.0]))
        assert result.mahalanobis_distance == pytest.approx(2.0, abs=1e-10)

    def test_invalid_mean_shape(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            OODDetectorWrapper(
                mean=np.array([[0.0, 0.0]]),
                precision_matrix=np.eye(2),
            )

    def test_mismatched_dimensions(self):
        with pytest.raises(ValueError, match="does not match"):
            OODDetectorWrapper(
                mean=np.array([0.0, 0.0, 0.0]),
                precision_matrix=np.eye(2),
            )


class TestFromNpz:
    def test_load_and_score(self, tmp_path):
        """Test loading from .npz file."""
        mean = np.array([1.0, 2.0, 3.0])
        precision = np.eye(3) * 0.5
        npz_path = tmp_path / "ood_params.npz"
        np.savez(npz_path, mean=mean, precision_matrix=precision, threshold=42.0)

        detector = OODDetectorWrapper.from_npz(str(npz_path))
        assert detector.threshold == 42.0

        result = detector.score(mean)
        assert result.mahalanobis_distance == pytest.approx(0.0, abs=1e-10)

    def test_override_threshold(self, tmp_path):
        """Explicit threshold overrides stored value."""
        npz_path = tmp_path / "ood_params.npz"
        np.savez(npz_path, mean=np.zeros(3), precision_matrix=np.eye(3), threshold=42.0)

        detector = OODDetectorWrapper.from_npz(str(npz_path), threshold=99.0)
        assert detector.threshold == 99.0
