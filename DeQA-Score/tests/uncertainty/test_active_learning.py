"""Tests for active_learning.py."""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.uncertainty.active_learning import (
    ActiveLearningSelector,
    ActiveLearningSample,
)
from src.uncertainty.fusion import (
    AcceptanceDecision,
    AcceptanceTier,
    UncertaintySignals,
)
from src.uncertainty.pseudo_label import PseudoLabelSample


def _make_sample(
    image_id: str,
    dimension: str = "overall",
    level_probs: np.ndarray | None = None,
    mos: float = 3.0,
) -> PseudoLabelSample:
    """Helper to create a PseudoLabelSample."""
    if level_probs is None:
        level_probs = np.array([0.1, 0.3, 0.4, 0.15, 0.05])
    dummy_signals = UncertaintySignals(
        mahalanobis_distance=30.0,
        cross_model_jsd=0.03,
        siglip2_sigma_sq=0.5,
        siglip2_entropy=1.0,
    )
    decision = AcceptanceDecision(
        image_id=image_id,
        dimension=dimension,
        tier=AcceptanceTier.AUTO_ACCEPT,
        confidence_weight=1.0,
        signals=dummy_signals,
        reason="test",
    )
    return PseudoLabelSample(
        image_id=image_id,
        dimension=dimension,
        level_probs=level_probs,
        mos=mos,
        std=0.5,
        confidence_weight=1.0,
        tier=AcceptanceTier.AUTO_ACCEPT,
        decision=decision,
    )


class TestScoreSamples:
    def test_basic_scoring(self):
        selector = ActiveLearningSelector()
        samples = [
            _make_sample("img001.jpg"),
            _make_sample("img002.jpg"),
        ]
        scored = selector.score_samples(samples)
        assert len(scored) == 2
        assert all(isinstance(s, ActiveLearningSample) for s in scored)

    def test_sorted_by_bald_descending(self):
        selector = ActiveLearningSelector()
        # Create samples with different entropy profiles
        samples = [
            _make_sample("img001.jpg", level_probs=np.array([1.0, 0.0, 0.0, 0.0, 0.0])),
            _make_sample("img002.jpg", level_probs=np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
        ]
        scored = selector.score_samples(samples)
        # Without ensemble, BALD=0 for all (single distribution)
        assert all(s.bald == 0.0 for s in scored)

    def test_bald_with_deqa_ensemble(self):
        selector = ActiveLearningSelector()
        # SigLIP2 says excellent, DeQA says bad → high BALD
        samples = [
            _make_sample("img001.jpg", level_probs=np.array([0.9, 0.1, 0.0, 0.0, 0.0])),
        ]
        deqa_probs = {
            "overall": {
                "img001.jpg": np.array([0.0, 0.0, 0.0, 0.1, 0.9]),
            }
        }
        scored = selector.score_samples(samples, deqa_probs_lookup=deqa_probs)
        assert len(scored) == 1
        assert scored[0].bald > 0.1  # Significant disagreement

    def test_sacred_ids_excluded(self):
        selector = ActiveLearningSelector(sacred_test_ids={"img001.jpg"})
        samples = [
            _make_sample("img001.jpg"),
            _make_sample("img002.jpg"),
        ]
        scored = selector.score_samples(samples)
        assert len(scored) == 1
        assert scored[0].image_id == "img002.jpg"

    def test_entropy_populated(self):
        selector = ActiveLearningSelector()
        samples = [
            _make_sample("img001.jpg", level_probs=np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        ]
        scored = selector.score_samples(samples)
        assert scored[0].entropy > 1.5  # Near max entropy


class TestSelectBatch:
    def test_selects_top_k(self):
        selector = ActiveLearningSelector()
        scored = [
            ActiveLearningSample(
                "img001.jpg", "overall", bald=0.5, entropy=1.0, mos=3.0
            ),
            ActiveLearningSample(
                "img002.jpg", "overall", bald=0.3, entropy=0.8, mos=3.5
            ),
            ActiveLearningSample(
                "img003.jpg", "overall", bald=0.1, entropy=0.5, mos=4.0
            ),
        ]
        selected = selector.select_batch(scored, k=2)
        assert len(selected) == 2
        assert selected[0].image_id == "img001.jpg"
        assert selected[1].image_id == "img002.jpg"

    def test_excludes_already_labeled(self):
        selector = ActiveLearningSelector()
        scored = [
            ActiveLearningSample(
                "img001.jpg", "overall", bald=0.5, entropy=1.0, mos=3.0
            ),
            ActiveLearningSample(
                "img002.jpg", "overall", bald=0.3, entropy=0.8, mos=3.5
            ),
        ]
        selected = selector.select_batch(scored, already_labeled={"img001.jpg"}, k=5)
        assert len(selected) == 1
        assert selected[0].image_id == "img002.jpg"

    def test_excludes_sacred_ids(self):
        selector = ActiveLearningSelector(sacred_test_ids={"img002.jpg"})
        scored = [
            ActiveLearningSample(
                "img001.jpg", "overall", bald=0.5, entropy=1.0, mos=3.0
            ),
            ActiveLearningSample(
                "img002.jpg", "overall", bald=0.3, entropy=0.8, mos=3.5
            ),
        ]
        selected = selector.select_batch(scored, k=5)
        assert len(selected) == 1

    def test_returns_fewer_than_k_if_insufficient(self):
        selector = ActiveLearningSelector()
        scored = [
            ActiveLearningSample(
                "img001.jpg", "overall", bald=0.5, entropy=1.0, mos=3.0
            ),
        ]
        selected = selector.select_batch(scored, k=100)
        assert len(selected) == 1


class TestGenerateAnnotationQueue:
    def test_writes_json(self):
        selector = ActiveLearningSelector()
        selected = [
            ActiveLearningSample(
                "img001.jpg", "overall", bald=0.5, entropy=1.0, mos=3.0
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "queue.json"
            n = selector.generate_annotation_queue(selected, output)
            assert n == 1
            with open(output) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["image_id"] == "img001.jpg"
            assert "bald_score" in data[0]
            assert "entropy" in data[0]


class TestFromSacredIdsFile:
    def test_loads_ids(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["test001.jpg", "test002.jpg"], f)
            f.flush()
            selector = ActiveLearningSelector.from_sacred_ids_file(f.name)
        assert len(selector.sacred_test_ids) == 2
        assert "test001.jpg" in selector.sacred_test_ids


class TestConvergence:
    def test_not_converged_with_one_iteration(self):
        metrics = [
            {
                "iteration": 1,
                "srcc": 0.85,
                "plcc": 0.82,
                "mean_bald": 0.1,
                "samples_above_threshold": 50,
            }
        ]
        result = ActiveLearningSelector.check_convergence(metrics)
        assert not result.is_converged
        assert result.srcc_delta == 1.0

    def test_converged(self):
        metrics = [
            {
                "iteration": 1,
                "srcc": 0.85,
                "plcc": 0.82,
                "mean_bald": 0.1,
                "samples_above_threshold": 50,
            },
            {
                "iteration": 2,
                "srcc": 0.856,
                "plcc": 0.83,
                "mean_bald": 0.05,
                "samples_above_threshold": 30,
            },
        ]
        result = ActiveLearningSelector.check_convergence(metrics)
        assert result.is_converged
        assert abs(result.srcc_delta - 0.006) < 1e-10

    def test_not_converged(self):
        metrics = [
            {
                "iteration": 1,
                "srcc": 0.80,
                "plcc": 0.78,
                "mean_bald": 0.2,
                "samples_above_threshold": 100,
            },
            {
                "iteration": 2,
                "srcc": 0.86,
                "plcc": 0.84,
                "mean_bald": 0.1,
                "samples_above_threshold": 50,
            },
        ]
        result = ActiveLearningSelector.check_convergence(metrics)
        assert not result.is_converged
        assert result.srcc_delta > 0.01
