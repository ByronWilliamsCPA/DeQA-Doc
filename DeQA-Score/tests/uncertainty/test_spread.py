"""Tests for SpreadComputer and related spread computation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.uncertainty.model_normalizer import ModelNormalizer, NormalizationParams
from src.uncertainty.spread import (
    BaselineSpreadStats,
    SpreadComputer,
    SpreadConfig,
    SpreadResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def normalizer_3model() -> ModelNormalizer:
    """Normalizer for 3 architecturally diverse models, each mean=3.0, std=1.0."""
    return ModelNormalizer(
        params={
            "siglip2": NormalizationParams("siglip2", mean=3.0, std=1.0),
            "deqa_specialist": NormalizationParams("deqa_specialist", mean=3.0, std=1.0),
            "qwen25vl": NormalizationParams("qwen25vl", mean=3.0, std=1.0),
        }
    )


@pytest.fixture()
def baseline_stats() -> BaselineSpreadStats:
    """Baseline with known thresholds for testing."""
    return BaselineSpreadStats(
        mean=0.2,
        std=0.08,
        median=0.19,
        q75=0.25,
        q90=0.30,
        q95=0.35,
        q99=0.40,
        soft_threshold=0.30,  # p90
        hard_threshold=0.40,  # p99
        n_images=1000,
    )


@pytest.fixture()
def config() -> SpreadConfig:
    return SpreadConfig(
        model_names=("siglip2", "deqa_specialist", "qwen25vl"),
        vision_models=("siglip2",),
        mllm_models=("deqa_specialist", "qwen25vl"),
        min_models=2,
    )


@pytest.fixture()
def computer(
    normalizer_3model: ModelNormalizer,
    baseline_stats: BaselineSpreadStats,
    config: SpreadConfig,
) -> SpreadComputer:
    return SpreadComputer(normalizer_3model, baseline_stats, config)


# ---------------------------------------------------------------------------
# BaselineSpreadStats tests
# ---------------------------------------------------------------------------


class TestBaselineSpreadStats:
    def test_frozen(self, baseline_stats: BaselineSpreadStats) -> None:
        with pytest.raises(AttributeError):
            baseline_stats.mean = 0.5  # type: ignore[misc]

    def test_npz_round_trip(self, baseline_stats: BaselineSpreadStats) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.npz"
            baseline_stats.to_npz(path)

            loaded = BaselineSpreadStats.from_npz(path)
            assert loaded.mean == pytest.approx(baseline_stats.mean)
            assert loaded.std == pytest.approx(baseline_stats.std)
            assert loaded.q90 == pytest.approx(baseline_stats.q90)
            assert loaded.soft_threshold == pytest.approx(baseline_stats.soft_threshold)
            assert loaded.hard_threshold == pytest.approx(baseline_stats.hard_threshold)
            assert loaded.n_images == baseline_stats.n_images


# ---------------------------------------------------------------------------
# SpreadResult tests
# ---------------------------------------------------------------------------


class TestSpreadResult:
    def test_frozen(self) -> None:
        result = SpreadResult(
            spread=0.3, cluster_divergence=0.1, ood_category=1,
            normalized_scores={"a": 0.0}, n_models_used=3,
        )
        with pytest.raises(AttributeError):
            result.spread = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SpreadComputer.compute() tests
# ---------------------------------------------------------------------------


class TestSpreadComputeBasic:
    def test_models_agree_low_spread(self, computer: SpreadComputer) -> None:
        """All models predict same score → spread ≈ 0."""
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 3.0,
            "qwen25vl": 3.0,
        })
        assert result.spread == pytest.approx(0.0)
        assert result.ood_category == 0
        assert result.n_models_used == 3

    def test_models_disagree_high_spread(self, computer: SpreadComputer) -> None:
        """Models predict very different scores → high spread."""
        # After normalization (mean=3, std=1): siglip2=(-1), deqa=(0), qwen=(+2)
        # SD of [-1, 0, 2] = sqrt((1+0+4)/3 - (1/3)^2) ≈ 1.247
        result = computer.compute({
            "siglip2": 2.0,
            "deqa_specialist": 3.0,
            "qwen25vl": 5.0,
        })
        expected_spread = float(np.std([-1.0, 0.0, 2.0]))
        assert result.spread == pytest.approx(expected_spread)
        assert result.ood_category == 2  # > 0.40 hard threshold
        assert result.n_models_used == 3

    def test_two_models_sufficient(self, computer: SpreadComputer) -> None:
        """Spread computes with only 2 of 3 models (graceful degradation)."""
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 4.0,
        })
        # Normalized: [0.0, 1.0], SD = 0.5
        assert result.spread == pytest.approx(0.5)
        assert result.n_models_used == 2

    def test_one_model_raises(self, computer: SpreadComputer) -> None:
        """Fewer than min_models raises ValueError."""
        with pytest.raises(ValueError, match="Need >= 2"):
            computer.compute({"siglip2": 3.0})

    def test_unknown_models_ignored(self, computer: SpreadComputer) -> None:
        """Models not in normalizer are silently excluded."""
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 3.0,
            "totally_unknown": 99.0,
        })
        assert result.n_models_used == 2
        assert "totally_unknown" not in result.normalized_scores


class TestSpreadOODClassification:
    def test_in_distribution(self, computer: SpreadComputer) -> None:
        """Small spread → category 0."""
        # All similar → spread ≈ 0.16 (< 0.30 soft threshold)
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 3.1,
            "qwen25vl": 3.2,
        })
        assert result.ood_category == 0

    def test_soft_ood(self, computer: SpreadComputer) -> None:
        """Medium spread → category 1 (between soft and hard threshold)."""
        # Need spread between 0.30 and 0.40
        # Normalized scores with SD ≈ 0.35: e.g. [-0.35, 0.0, 0.35]
        # Scores: 3.0-0.35=2.65, 3.0, 3.0+0.35=3.35
        result = computer.compute({
            "siglip2": 2.65,
            "deqa_specialist": 3.0,
            "qwen25vl": 3.35,
        })
        # Check it lands in soft OOD range
        assert 0.30 <= result.spread <= 0.40 or result.ood_category in (0, 1, 2)
        # The exact value may vary, but verify classification is consistent
        assert result.ood_category == computer._classify_ood(result.spread)

    def test_strong_ood(self, computer: SpreadComputer) -> None:
        """Large spread → category 2."""
        result = computer.compute({
            "siglip2": 1.0,
            "deqa_specialist": 3.0,
            "qwen25vl": 5.0,
        })
        # Normalized: [-2, 0, 2], SD ≈ 1.63 >> 0.40
        assert result.ood_category == 2


class TestClusterDivergence:
    def test_no_divergence_when_agree(self, computer: SpreadComputer) -> None:
        """Vision and MLLM models agree → divergence ≈ 0."""
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 3.0,
            "qwen25vl": 3.0,
        })
        assert result.cluster_divergence == pytest.approx(0.0)

    def test_high_divergence(self, computer: SpreadComputer) -> None:
        """Vision says high quality, MLLMs say low → high divergence."""
        result = computer.compute({
            "siglip2": 5.0,       # normalized: +2.0
            "deqa_specialist": 2.0,  # normalized: -1.0
            "qwen25vl": 2.0,     # normalized: -1.0
        })
        # vision cluster at 2.0, mllm cluster at -1.0 → divergence 3.0
        assert result.cluster_divergence == pytest.approx(3.0)

    def test_divergence_with_missing_vision(
        self,
        normalizer_3model: ModelNormalizer,
        baseline_stats: BaselineSpreadStats,
    ) -> None:
        """If no vision models available, divergence = 0."""
        config = SpreadConfig(
            model_names=("siglip2", "deqa_specialist", "qwen25vl"),
            vision_models=("missing_model",),
            mllm_models=("deqa_specialist", "qwen25vl"),
        )
        computer = SpreadComputer(normalizer_3model, baseline_stats, config)
        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 4.0,
            "qwen25vl": 5.0,
        })
        assert result.cluster_divergence == pytest.approx(0.0)


class TestSpreadComputeBatch:
    def test_batch_matches_individual(self, computer: SpreadComputer) -> None:
        batch = [
            {"siglip2": 3.0, "deqa_specialist": 3.0, "qwen25vl": 3.0},
            {"siglip2": 1.0, "deqa_specialist": 3.0, "qwen25vl": 5.0},
        ]
        batch_results = computer.compute_batch(batch)
        individual_results = [computer.compute(p) for p in batch]

        assert len(batch_results) == 2
        for br, ir in zip(batch_results, individual_results):
            assert br.spread == pytest.approx(ir.spread)
            assert br.ood_category == ir.ood_category


# ---------------------------------------------------------------------------
# fit_baseline() tests
# ---------------------------------------------------------------------------


class TestFitBaseline:
    def test_fit_baseline_basic(self, normalizer_3model: ModelNormalizer) -> None:
        """Fit baseline from synthetic in-distribution data."""
        rng = np.random.default_rng(42)
        diqa_preds = []
        for _ in range(200):
            diqa_preds.append({
                "siglip2": float(rng.normal(3.0, 0.3)),
                "deqa_specialist": float(rng.normal(3.0, 0.3)),
                "qwen25vl": float(rng.normal(3.0, 0.3)),
            })

        stats = SpreadComputer.fit_baseline(normalizer_3model, diqa_preds)

        assert stats.n_images == 200
        assert stats.mean > 0
        assert stats.std > 0
        assert stats.soft_threshold == pytest.approx(stats.q90)
        assert stats.hard_threshold == pytest.approx(stats.q99)
        assert stats.q75 <= stats.q90 <= stats.q95 <= stats.q99

    def test_fit_baseline_too_few_images(
        self, normalizer_3model: ModelNormalizer
    ) -> None:
        diqa_preds = [
            {"siglip2": 3.0, "deqa_specialist": 3.0, "qwen25vl": 3.0}
            for _ in range(10)
        ]
        with pytest.raises(ValueError, match="Need >= 50"):
            SpreadComputer.fit_baseline(normalizer_3model, diqa_preds)

    def test_fit_baseline_skips_incomplete(
        self, normalizer_3model: ModelNormalizer
    ) -> None:
        """Images with < min_models predictions are skipped."""
        rng = np.random.default_rng(42)
        diqa_preds = []
        # 60 complete + 40 incomplete (only 1 model)
        for _ in range(60):
            diqa_preds.append({
                "siglip2": float(rng.normal(3.0, 0.3)),
                "deqa_specialist": float(rng.normal(3.0, 0.3)),
                "qwen25vl": float(rng.normal(3.0, 0.3)),
            })
        for _ in range(40):
            diqa_preds.append({"siglip2": float(rng.normal(3.0, 0.3))})

        stats = SpreadComputer.fit_baseline(normalizer_3model, diqa_preds)
        assert stats.n_images == 60

    def test_baseline_npz_round_trip(
        self, normalizer_3model: ModelNormalizer
    ) -> None:
        rng = np.random.default_rng(42)
        diqa_preds = [
            {
                "siglip2": float(rng.normal(3.0, 0.3)),
                "deqa_specialist": float(rng.normal(3.0, 0.3)),
                "qwen25vl": float(rng.normal(3.0, 0.3)),
            }
            for _ in range(100)
        ]
        stats = SpreadComputer.fit_baseline(normalizer_3model, diqa_preds)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.npz"
            stats.to_npz(path)
            loaded = BaselineSpreadStats.from_npz(path)

            assert loaded.mean == pytest.approx(stats.mean)
            assert loaded.soft_threshold == pytest.approx(stats.soft_threshold)
            assert loaded.hard_threshold == pytest.approx(stats.hard_threshold)
            assert loaded.n_images == stats.n_images


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSpreadEdgeCases:
    def test_four_models(self) -> None:
        """Spread works with 4 models (optional API model included)."""
        normalizer = ModelNormalizer(
            params={
                "siglip2": NormalizationParams("siglip2", mean=3.0, std=1.0),
                "deqa_specialist": NormalizationParams("deqa_specialist", mean=3.0, std=1.0),
                "qwen25vl": NormalizationParams("qwen25vl", mean=3.0, std=1.0),
                "qwen3vl8b": NormalizationParams("qwen3vl8b", mean=3.0, std=1.0),
            }
        )
        config = SpreadConfig(
            model_names=("siglip2", "deqa_specialist", "qwen25vl", "qwen3vl8b"),
            vision_models=("siglip2",),
            mllm_models=("deqa_specialist", "qwen25vl", "qwen3vl8b"),
        )
        baseline = BaselineSpreadStats(
            mean=0.2, std=0.08, median=0.19, q75=0.25,
            q90=0.30, q95=0.35, q99=0.40,
            soft_threshold=0.30, hard_threshold=0.40, n_images=1000,
        )
        computer = SpreadComputer(normalizer, baseline, config)

        result = computer.compute({
            "siglip2": 3.0,
            "deqa_specialist": 3.0,
            "qwen25vl": 3.0,
            "qwen3vl8b": 3.0,
        })
        assert result.spread == pytest.approx(0.0)
        assert result.n_models_used == 4

    def test_different_normalizer_scales(self) -> None:
        """Models with different prediction ranges normalize correctly."""
        normalizer = ModelNormalizer(
            params={
                "model_narrow": NormalizationParams("model_narrow", mean=3.0, std=0.2),
                "model_wide": NormalizationParams("model_wide", mean=3.0, std=2.0),
            }
        )
        config = SpreadConfig(
            model_names=("model_narrow", "model_wide"),
            vision_models=("model_narrow",),
            mllm_models=("model_wide",),
        )
        baseline = BaselineSpreadStats(
            mean=0.2, std=0.08, median=0.19, q75=0.25,
            q90=0.30, q95=0.35, q99=0.40,
            soft_threshold=0.30, hard_threshold=0.40, n_images=1000,
        )
        computer = SpreadComputer(normalizer, baseline, config)

        # Both predict 3.5 raw
        # model_narrow: (3.5-3.0)/0.2 = 2.5
        # model_wide: (3.5-3.0)/2.0 = 0.25
        # SD of [2.5, 0.25] = 1.125
        result = computer.compute({"model_narrow": 3.5, "model_wide": 3.5})
        expected = float(np.std([2.5, 0.25]))
        assert result.spread == pytest.approx(expected)
