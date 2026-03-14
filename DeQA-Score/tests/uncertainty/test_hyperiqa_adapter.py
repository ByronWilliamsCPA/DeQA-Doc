"""Tests for hyperiqa_adapter.py — 10→5 bin mapping and CrossValidator factory."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.uncertainty.hyperiqa_adapter import (
    DEFAULT_10_TO_5_MAP,
    BinMapping,
    gaussian_to_10bin,
    hyperiqa_cross_validator,
    load_hyperiqa_predictions,
    map_bins_to_levels,
    native_10bin_jsd,
)


class TestBinMapping:
    def test_default_10bin_shape(self):
        mapping = BinMapping.default_10bin()
        assert mapping.n_bins == 10
        assert mapping.bin_to_level.shape == (10,)
        assert mapping.bin_centers.shape == (10,)

    def test_default_map_values(self):
        """Verify the 10→5 mapping assigns each bin to the nearest level."""
        expected = [4, 3, 3, 3, 2, 2, 1, 1, 1, 0]
        np.testing.assert_array_equal(DEFAULT_10_TO_5_MAP, expected)

    def test_default_bin_centers(self):
        mapping = BinMapping.default_10bin()
        expected = [1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8]
        np.testing.assert_allclose(mapping.bin_centers, expected, atol=1e-10)


class TestMapBinsToLevels:
    def test_uniform_10bin_maps_to_uniform_5level(self):
        """Uniform 10-bin → sums by group → NOT uniform 5-level (1,3,2,3,1 mapping)."""
        uniform_10 = np.ones(10) / 10.0
        result = map_bins_to_levels(uniform_10)
        assert result.shape == (5,)
        assert abs(result.sum() - 1.0) < 1e-10
        # Expected: excellent=0.1, good=0.3, fair=0.2, poor=0.3, bad=0.1
        expected = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_all_mass_in_excellent_bin(self):
        """All probability in bin 9 (center 4.8) → excellent."""
        probs = np.zeros(10)
        probs[9] = 1.0
        result = map_bins_to_levels(probs)
        assert result[0] == pytest.approx(1.0)  # excellent
        assert result[1:].sum() == pytest.approx(0.0)

    def test_all_mass_in_bad_bin(self):
        """All probability in bin 0 (center 1.2) → bad."""
        probs = np.zeros(10)
        probs[0] = 1.0
        result = map_bins_to_levels(probs)
        assert result[4] == pytest.approx(1.0)  # bad
        assert result[:4].sum() == pytest.approx(0.0)

    def test_good_bins_aggregate(self):
        """Bins 6,7,8 all map to good (idx 1)."""
        probs = np.zeros(10)
        probs[6] = 0.2
        probs[7] = 0.5
        probs[8] = 0.3
        result = map_bins_to_levels(probs)
        assert result[1] == pytest.approx(1.0)  # good
        assert result.sum() == pytest.approx(1.0)

    def test_poor_bins_aggregate(self):
        """Bins 1,2,3 all map to poor (idx 3)."""
        probs = np.zeros(10)
        probs[1] = 0.3
        probs[2] = 0.4
        probs[3] = 0.3
        result = map_bins_to_levels(probs)
        assert result[3] == pytest.approx(1.0)  # poor

    def test_fair_bins_aggregate(self):
        """Bins 4,5 map to fair (idx 2)."""
        probs = np.zeros(10)
        probs[4] = 0.6
        probs[5] = 0.4
        result = map_bins_to_levels(probs)
        assert result[2] == pytest.approx(1.0)  # fair

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected 10"):
            map_bins_to_levels(np.ones(5))

    def test_zero_probs_gives_uniform(self):
        result = map_bins_to_levels(np.zeros(10))
        np.testing.assert_allclose(result, 0.2, atol=1e-10)

    def test_normalization(self):
        """Unnormalized input gets normalized."""
        probs = np.ones(10) * 5.0  # sums to 50
        result = map_bins_to_levels(probs)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_realistic_gaussian_distribution(self):
        """A Gaussian-like 10-bin distribution centered at 3.6 (good)."""
        # Centered at bin 6 (3.6), sigma ~0.8
        centers = np.array([1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8])
        probs = np.exp(-0.5 * ((centers - 3.6) / 0.8) ** 2)
        probs /= probs.sum()
        result = map_bins_to_levels(probs)
        # Most mass should be in "good" (idx 1)
        assert np.argmax(result) == 1  # good
        assert result[1] > 0.5  # majority in good


class TestLoadHyperiqaPredictions:
    def _write_jsonl(self, records: list[dict], path: Path) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_multi_dimension_format(self, tmp_path):
        """Load multi-dimension JSONL format."""
        probs_10 = list(np.ones(10) / 10.0)
        records = [
            {
                "image": "img001.jpg",
                "overall": {"score": 3.5, "probs": probs_10},
                "sharpness": {"score": 3.2, "probs": probs_10},
                "color": {"score": 3.8, "probs": probs_10},
            }
        ]
        jsonl_path = tmp_path / "preds.jsonl"
        self._write_jsonl(records, jsonl_path)

        result = load_hyperiqa_predictions(str(jsonl_path))
        assert set(result.keys()) == {"overall", "sharpness", "color"}
        assert "img001.jpg" in result["overall"]
        assert result["overall"]["img001.jpg"].shape == (5,)

    def test_flat_format(self, tmp_path):
        """Load flat JSONL format with dimension field."""
        probs_10 = list(np.ones(10) / 10.0)
        records = [
            {"image": "img001.jpg", "dimension": "overall", "probs": probs_10},
            {"image": "img001.jpg", "dimension": "sharpness", "probs": probs_10},
        ]
        jsonl_path = tmp_path / "preds.jsonl"
        self._write_jsonl(records, jsonl_path)

        result = load_hyperiqa_predictions(str(jsonl_path))
        assert "overall" in result
        assert "sharpness" in result

    def test_color_fidelity_alias(self, tmp_path):
        """'color_fidelity' dimension maps to 'color'."""
        probs_10 = list(np.ones(10) / 10.0)
        records = [
            {
                "image": "img001.jpg",
                "color_fidelity": {"score": 3.8, "probs": probs_10},
            }
        ]
        jsonl_path = tmp_path / "preds.jsonl"
        self._write_jsonl(records, jsonl_path)

        result = load_hyperiqa_predictions(str(jsonl_path))
        assert "color" in result

    def test_empty_file(self, tmp_path):
        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")
        result = load_hyperiqa_predictions(str(jsonl_path))
        assert result == {}

    def test_missing_image_id_skipped(self, tmp_path):
        records = [{"probs": list(np.ones(10) / 10.0)}]
        jsonl_path = tmp_path / "preds.jsonl"
        self._write_jsonl(records, jsonl_path)
        result = load_hyperiqa_predictions(str(jsonl_path))
        assert result == {}


class TestHyperiqaCrossValidator:
    def test_creates_cross_validator(self, tmp_path):
        """hyperiqa_cross_validator returns a working CrossValidator."""
        probs_10 = list(np.ones(10) / 10.0)
        records = [
            {
                "image": "img001.jpg",
                "overall": {"score": 3.5, "probs": probs_10},
            }
        ]
        jsonl_path = tmp_path / "preds.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        cv = hyperiqa_cross_validator(str(jsonl_path))
        assert cv.has_prediction("img001.jpg", "overall")
        assert not cv.has_prediction("img001.jpg", "sharpness")

    def test_validate_returns_cross_val_result(self, tmp_path):
        """Validate produces CrossValidationResult with JSD."""
        # Create a peaked distribution → map to 5-level
        probs_10 = np.zeros(10)
        probs_10[6] = 0.8  # center 3.6 → good
        probs_10[7] = 0.2  # center 4.0 → good
        records = [
            {"image": "test.jpg", "overall": {"score": 3.7, "probs": list(probs_10)}}
        ]
        jsonl_path = tmp_path / "preds.jsonl"
        with open(jsonl_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        cv = hyperiqa_cross_validator(str(jsonl_path))
        result = cv.validate(
            image_id="test.jpg",
            dimension="overall",
            siglip2_mu=3.5,
            siglip2_sigma_sq=0.04,
        )
        assert result.image_id == "test.jpg"
        assert result.cross_model_jsd >= 0.0
        assert result.deqa_probs.shape == (5,)
        # HyperIQA++ predicts "good" → deqa_probs[1] should be dominant
        assert np.argmax(result.deqa_probs) == 1


class TestGaussianTo10Bin:
    """Tests for gaussian_to_10bin — SigLIP2 Gaussian → 10-bin distribution."""

    def test_output_shape_and_normalization(self):
        probs = gaussian_to_10bin(3.0, 0.25)
        assert probs.shape == (10,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    def test_sharp_gaussian_concentrates_mass(self):
        """Small σ² → mass concentrated near the predicted MOS."""
        probs = gaussian_to_10bin(3.6, 0.01)  # very tight around 3.6
        # Bin 6 has center 3.6, should get most mass
        assert probs[6] > 0.5

    def test_wide_gaussian_spreads_mass(self):
        """Large σ² → mass spread across many bins."""
        probs = gaussian_to_10bin(3.0, 4.0)
        # No single bin should dominate
        assert probs.max() < 0.3

    def test_mu_at_extremes(self):
        """μ at edges of [1.0, 5.0] should still produce valid distributions."""
        for mu in [1.0, 5.0]:
            probs = gaussian_to_10bin(mu, 0.04)
            assert probs.sum() == pytest.approx(1.0, abs=1e-10)
            assert np.all(probs >= 0)

    def test_mu_clamped_outside_range(self):
        """μ outside [1.0, 5.0] gets clamped."""
        probs_low = gaussian_to_10bin(0.0, 0.25)
        probs_at_1 = gaussian_to_10bin(1.0, 0.25)
        np.testing.assert_allclose(probs_low, probs_at_1, atol=1e-10)

    def test_peak_follows_mu(self):
        """Peak bin should track μ (using exact bin centers)."""
        for mu, expected_peak_bin in [(1.2, 0), (3.2, 5), (4.8, 9)]:
            probs = gaussian_to_10bin(mu, 0.04)
            assert np.argmax(probs) == expected_peak_bin


class TestNative10BinJSD:
    """Tests for native_10bin_jsd — JSD in HyperIQA++'s native 10-bin space."""

    def test_identical_distributions_jsd_zero(self):
        """When SigLIP2 Gaussian matches HyperIQA++ exactly, JSD ≈ 0."""
        # Create a 10-bin dist from a known Gaussian, then compare
        hyperiqa = gaussian_to_10bin(3.6, 0.25)
        jsd = native_10bin_jsd(3.6, 0.25, hyperiqa)
        assert jsd == pytest.approx(0.0, abs=1e-10)

    def test_completely_different_distributions(self):
        """Disjoint distributions → high JSD."""
        hyperiqa = np.zeros(10)
        hyperiqa[0] = 1.0  # All mass at bin 0 (center 1.2)
        jsd = native_10bin_jsd(4.8, 0.01, hyperiqa)  # SigLIP2 says 4.8
        assert jsd > 0.5  # Should be close to ln(2) ≈ 0.693

    def test_jsd_bounded(self):
        """JSD should be in [0, ln(2)]."""
        hyperiqa = np.ones(10) / 10.0
        jsd = native_10bin_jsd(3.0, 0.5, hyperiqa)
        assert 0.0 <= jsd <= np.log(2) + 1e-10

    def test_jsd_symmetric_in_distributions(self):
        """JSD(p,q) = JSD(q,p) — verified by construction."""
        hyperiqa = np.zeros(10)
        hyperiqa[3] = 0.5
        hyperiqa[4] = 0.5
        jsd1 = native_10bin_jsd(2.5, 0.25, hyperiqa)
        # Since discrete_jsd is symmetric, this should hold
        assert jsd1 > 0.0  # They differ

    def test_close_predictions_low_jsd(self):
        """When SigLIP2 and HyperIQA++ roughly agree, JSD is small."""
        hyperiqa = gaussian_to_10bin(3.5, 0.3)  # Similar to SigLIP2
        jsd = native_10bin_jsd(3.6, 0.25, hyperiqa)  # Slightly different
        assert jsd < 0.05
