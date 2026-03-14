"""Tests for ModelNormalizer z-score normalization."""

from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.model_normalizer import ModelNormalizer, NormalizationParams


class TestNormalizationParams:
    """Tests for the NormalizationParams frozen dataclass."""

    def test_frozen(self) -> None:
        params = NormalizationParams(model_name="test", mean=3.0, std=0.5)
        with pytest.raises(AttributeError):
            params.mean = 4.0  # type: ignore[misc]

    def test_fields(self) -> None:
        params = NormalizationParams(model_name="siglip2", mean=3.2, std=0.8)
        assert params.model_name == "siglip2"
        assert params.mean == pytest.approx(3.2)
        assert params.std == pytest.approx(0.8)


class TestModelNormalizerFit:
    """Tests for fitting normalization parameters."""

    def test_fit_basic(self) -> None:
        normalizer = ModelNormalizer()
        predictions = {
            "model_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            "model_b": [2.0, 3.0, 4.0, 5.0, 6.0] * 10,
        }
        normalizer.fit(predictions)

        assert normalizer.is_fitted
        assert set(normalizer.model_names) == {"model_a", "model_b"}

        params_a = normalizer.get_params("model_a")
        assert params_a.mean == pytest.approx(3.0)
        assert params_a.std == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_fit_rejects_too_few_predictions(self) -> None:
        normalizer = ModelNormalizer()
        with pytest.raises(ValueError, match="need >= 10"):
            normalizer.fit({"model_a": [1.0, 2.0, 3.0]})

    def test_fit_rejects_zero_variance(self) -> None:
        normalizer = ModelNormalizer()
        with pytest.raises(ValueError, match="zero variance"):
            normalizer.fit({"model_a": [3.0] * 20})

    def test_unfitted_raises(self) -> None:
        normalizer = ModelNormalizer()
        assert not normalizer.is_fitted
        with pytest.raises(KeyError, match="not fitted"):
            normalizer.get_params("anything")


class TestModelNormalizerTransform:
    """Tests for z-score transformation."""

    @pytest.fixture()
    def fitted_normalizer(self) -> ModelNormalizer:
        """Normalizer fitted with known mean=3.0, std=1.0 for model_a."""
        params = {
            "model_a": NormalizationParams("model_a", mean=3.0, std=1.0),
            "model_b": NormalizationParams("model_b", mean=4.0, std=0.5),
        }
        return ModelNormalizer(params=params)

    def test_transform_basic(self, fitted_normalizer: ModelNormalizer) -> None:
        # model_a: mean=3.0, std=1.0 → z-score of 3.0 is 0.0
        assert fitted_normalizer.transform("model_a", 3.0) == pytest.approx(0.0)
        # model_a: z-score of 5.0 is 2.0
        assert fitted_normalizer.transform("model_a", 5.0) == pytest.approx(2.0)
        # model_b: mean=4.0, std=0.5 → z-score of 4.0 is 0.0
        assert fitted_normalizer.transform("model_b", 4.0) == pytest.approx(0.0)
        # model_b: z-score of 5.0 is 2.0
        assert fitted_normalizer.transform("model_b", 5.0) == pytest.approx(2.0)

    def test_transform_unknown_model(self, fitted_normalizer: ModelNormalizer) -> None:
        with pytest.raises(KeyError, match="not fitted"):
            fitted_normalizer.transform("unknown", 3.0)

    def test_transform_batch(self, fitted_normalizer: ModelNormalizer) -> None:
        result = fitted_normalizer.transform_batch(
            {"model_a": 4.0, "model_b": 4.5}
        )
        assert result == pytest.approx({"model_a": 1.0, "model_b": 1.0})

    def test_transform_batch_skips_unfitted(
        self, fitted_normalizer: ModelNormalizer
    ) -> None:
        """Models in predictions but not in normalizer are silently skipped."""
        result = fitted_normalizer.transform_batch(
            {"model_a": 4.0, "unknown_model": 3.0}
        )
        assert "model_a" in result
        assert "unknown_model" not in result


class TestModelNormalizerSerialization:
    """Tests for .npz save/load round-trip."""

    def test_round_trip(self, tmp_path: object) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "normalizer.npz"

            original = ModelNormalizer(
                params={
                    "siglip2": NormalizationParams("siglip2", mean=3.2, std=0.8),
                    "deqa": NormalizationParams("deqa", mean=2.9, std=1.1),
                }
            )
            original.to_npz(path)

            loaded = ModelNormalizer.from_npz(path)
            assert set(loaded.model_names) == {"siglip2", "deqa"}

            p1 = loaded.get_params("siglip2")
            assert p1.mean == pytest.approx(3.2)
            assert p1.std == pytest.approx(0.8)

            p2 = loaded.get_params("deqa")
            assert p2.mean == pytest.approx(2.9)
            assert p2.std == pytest.approx(1.1)

    def test_save_unfitted_raises(self) -> None:
        normalizer = ModelNormalizer()
        with pytest.raises(ValueError, match="unfitted"):
            normalizer.to_npz("/tmp/should_not_exist.npz")

    def test_transform_matches_after_reload(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "norm.npz"

            original = ModelNormalizer()
            rng = np.random.default_rng(42)
            predictions = {
                "m1": rng.normal(3.0, 0.5, 100).tolist(),
                "m2": rng.normal(4.0, 1.0, 100).tolist(),
            }
            original.fit(predictions)

            score_before = original.transform("m1", 3.5)
            original.to_npz(path)

            loaded = ModelNormalizer.from_npz(path)
            score_after = loaded.transform("m1", 3.5)

            assert score_before == pytest.approx(score_after)
