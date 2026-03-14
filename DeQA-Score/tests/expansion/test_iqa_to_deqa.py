"""Tests for IQA-to-DeQA format conversion pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.expansion.iqa_to_deqa import (
    IQALabels,
    LEVEL_NAMES,
    iqa_to_deqa_record,
    mos_to_level_probs,
    overall_quality_to_mos,
    vlm_scores_to_deqa_record,
)
from src.uncertainty.gaussian_to_discrete import level_probs_to_mos


class TestOverallQualityToMOS:
    """Test overall_quality [0,1] → MOS [1,5] mapping."""

    def test_perfect_quality(self):
        assert overall_quality_to_mos(1.0) == 5.0

    def test_worst_quality(self):
        assert overall_quality_to_mos(0.0) == 1.0

    def test_midpoint(self):
        assert overall_quality_to_mos(0.5) == 3.0

    def test_clamping_above(self):
        assert overall_quality_to_mos(1.5) == 5.0

    def test_clamping_below(self):
        assert overall_quality_to_mos(-0.5) == 1.0

    @pytest.mark.parametrize("oq,expected_mos", [
        (0.0, 1.0),
        (0.25, 2.0),
        (0.5, 3.0),
        (0.75, 4.0),
        (1.0, 5.0),
    ])
    def test_linear_mapping(self, oq, expected_mos):
        assert overall_quality_to_mos(oq) == pytest.approx(expected_mos)


class TestMOSToLevelProbs:
    """Test MOS → 5-bin soft-label conversion."""

    def test_sums_to_one(self):
        for mos in [1.0, 2.0, 3.0, 4.0, 5.0]:
            probs = mos_to_level_probs(mos)
            assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_shape(self):
        probs = mos_to_level_probs(3.0)
        assert probs.shape == (5,)

    def test_excellent_peaks_at_index_0(self):
        probs = mos_to_level_probs(5.0)
        assert np.argmax(probs) == 0  # excellent

    def test_bad_peaks_at_index_4(self):
        probs = mos_to_level_probs(1.0)
        assert np.argmax(probs) == 4  # bad

    def test_fair_peaks_at_index_2(self):
        probs = mos_to_level_probs(3.0)
        assert np.argmax(probs) == 2  # fair

    def test_mos_roundtrip(self):
        """MOS → level_probs → reconstructed MOS should be close."""
        for mos in [1.5, 2.3, 3.0, 3.7, 4.5]:
            probs = mos_to_level_probs(mos, sigma=0.8)
            reconstructed = level_probs_to_mos(probs)
            assert reconstructed == pytest.approx(mos, abs=0.3)
        # Edge values (near 1 or 5) have larger error due to bin clipping
        for mos in [1.1, 4.8]:
            probs = mos_to_level_probs(mos, sigma=0.8)
            reconstructed = level_probs_to_mos(probs)
            assert reconstructed == pytest.approx(mos, abs=0.6)

    def test_binary_mode_for_small_sigma(self):
        probs = mos_to_level_probs(3.5, sigma=0.1)
        nonzero = np.count_nonzero(probs)
        assert nonzero <= 2


class TestIQAToDeqaRecord:
    """Test IQA labels → DeQA training record conversion."""

    def test_output_format(self):
        """Record must have all fields expected by SingleDataset."""
        iqa = IQALabels(overall_quality=0.7)
        record = iqa_to_deqa_record(
            image_id="test_001",
            image_path="test/test_001.png",
            iqa_labels=iqa,
            seed=42,
        )

        # Required fields
        assert "id" in record
        assert "image" in record
        assert "gt_score" in record
        assert "gt_score_norm" in record
        assert "level_probs" in record
        assert "conversations" in record
        assert "std" in record
        assert "std_norm" in record

        # Format checks
        assert len(record["level_probs"]) == 5
        assert sum(record["level_probs"]) == pytest.approx(1.0, abs=1e-4)
        assert len(record["conversations"]) == 2
        assert record["conversations"][0]["from"] == "human"
        assert record["conversations"][1]["from"] == "gpt"
        assert "<|image|>" in record["conversations"][0]["value"]

    def test_mos_range(self):
        """gt_score must be in [1, 5]."""
        for oq in [0.0, 0.3, 0.5, 0.7, 1.0]:
            iqa = IQALabels(overall_quality=oq)
            record = iqa_to_deqa_record(
                image_id="test", image_path="test.png", iqa_labels=iqa, seed=0
            )
            assert 1.0 <= record["gt_score"] <= 5.0

    def test_level_name_in_answer(self):
        """GPT answer must contain a valid quality level name."""
        iqa = IQALabels(overall_quality=0.8)
        record = iqa_to_deqa_record(
            image_id="test", image_path="test.png", iqa_labels=iqa, seed=42
        )
        answer = record["conversations"][1]["value"]
        assert any(level in answer for level in LEVEL_NAMES)

    def test_expansion_metadata(self):
        """Record should include expansion tracking fields."""
        iqa = IQALabels(overall_quality=0.5)
        record = iqa_to_deqa_record(
            image_id="test", image_path="test.png", iqa_labels=iqa,
            source="test_source", stream="stream1_degradation",
            weight=0.7, seed=42,
        )
        assert record["pseudo_label"] is True
        assert record["confidence_weight"] == 0.7
        assert record["source"] == "test_source"
        assert record["stream"] == "stream1_degradation"

    def test_dict_iqa_input(self):
        """Should accept dict as well as IQALabels dataclass."""
        iqa_dict = {"overall_quality": 0.6, "blur": 0.3}
        record = iqa_to_deqa_record(
            image_id="test", image_path="test.png", iqa_labels=iqa_dict, seed=42
        )
        expected_mos = 1.0 + 4.0 * 0.6  # 3.4
        assert record["gt_score"] == pytest.approx(expected_mos, abs=0.5)

    def test_deterministic_with_seed(self):
        """Same seed should produce same record."""
        iqa = IQALabels(overall_quality=0.5)
        r1 = iqa_to_deqa_record("t", "p", iqa, seed=42)
        r2 = iqa_to_deqa_record("t", "p", iqa, seed=42)
        assert r1["conversations"] == r2["conversations"]
        assert r1["level_probs"] == r2["level_probs"]


class TestVLMScoresToDeqaRecord:
    """Test VLM consensus MOS → DeQA record conversion."""

    def test_output_format(self):
        record = vlm_scores_to_deqa_record(
            image_id="vlm_001",
            image_path="test/vlm_001.png",
            vlm_mos=3.5,
            seed=42,
        )
        assert record["stream"] == "stream3_vlm"
        assert record["confidence_weight"] == 0.5
        assert len(record["level_probs"]) == 5

    def test_vlm_metadata(self):
        record = vlm_scores_to_deqa_record(
            image_id="vlm_001",
            image_path="test.png",
            vlm_mos=4.0,
            vlm_models=["model_a", "model_b"],
            seed=42,
        )
        assert "vlm_mos_raw" in record
        assert record["vlm_mos_raw"] == 4.0
        assert record["vlm_models"] == ["model_a", "model_b"]

    def test_clamped_mos(self):
        """MOS outside [1,5] should be clamped."""
        record = vlm_scores_to_deqa_record(
            image_id="x", image_path="x", vlm_mos=6.0, seed=0
        )
        assert record["gt_score"] <= 5.0

        record = vlm_scores_to_deqa_record(
            image_id="x", image_path="x", vlm_mos=-1.0, seed=0
        )
        assert record["gt_score"] >= 1.0
