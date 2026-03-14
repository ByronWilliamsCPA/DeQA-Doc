"""Tests for vlm_validator module.

Tests _parse_vlm_response edge cases, VLMBudgetTracker accounting,
VLMValidator.validate_single with mocked API, and select_tier2_queue capping.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.uncertainty.vlm_validator import (
    QUALITY_LEVEL_MAP,
    VLMBudgetTracker,
    VLMValidator,
    _parse_vlm_response,
)


# ---------------------------------------------------------------------------
# _parse_vlm_response
# ---------------------------------------------------------------------------

class TestParseVlmResponse:
    """Tests for _parse_vlm_response parsing logic."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("excellent", "excellent"),
            ("Good", "good"),
            ("FAIR", "fair"),
            ("Poor.", "poor"),
            ("  bad  ", "bad"),
            ("fair quality", "fair"),
            ("good - the image is clear", "good"),
        ],
    )
    def test_valid_levels(self, text: str, expected: str) -> None:
        assert _parse_vlm_response(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I think this image is great",
            "unknown",
            "3.5",
            "badly damaged",
            "goodness",
            "fairly",
        ],
    )
    def test_invalid_or_false_positive(self, text: str) -> None:
        assert _parse_vlm_response(text) is None

    def test_word_boundary_prevents_prefix_match(self) -> None:
        """Regression: 'badly' must not match 'bad'."""
        assert _parse_vlm_response("badly") is None
        assert _parse_vlm_response("bad") == "bad"

    def test_exact_match_all_levels(self) -> None:
        for level in QUALITY_LEVEL_MAP:
            assert _parse_vlm_response(level) == level


# ---------------------------------------------------------------------------
# VLMBudgetTracker
# ---------------------------------------------------------------------------

class TestVLMBudgetTracker:
    """Tests for budget accounting."""

    def test_initial_state(self) -> None:
        tracker = VLMBudgetTracker()
        assert tracker.total_calls == 0
        assert tracker.total_cost_usd == pytest.approx(0.0)
        assert tracker.vetoed_count == 0
        assert tracker.parse_failures == 0

    def test_record_call_increments(self) -> None:
        tracker = VLMBudgetTracker()
        tracker.record_call(vetoed=False, parse_success=True)
        assert tracker.total_calls == 1
        assert tracker.vetoed_count == 0
        assert tracker.parse_failures == 0
        assert tracker.total_cost_usd > 0.0

    def test_record_veto(self) -> None:
        tracker = VLMBudgetTracker()
        tracker.record_call(vetoed=True, parse_success=True)
        assert tracker.vetoed_count == 1

    def test_record_parse_failure(self) -> None:
        tracker = VLMBudgetTracker()
        tracker.record_call(vetoed=False, parse_success=False)
        assert tracker.parse_failures == 1

    def test_summary_veto_rate(self) -> None:
        tracker = VLMBudgetTracker()
        tracker.record_call(vetoed=True, parse_success=True)
        tracker.record_call(vetoed=False, parse_success=True)
        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["veto_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# VLMValidator.validate_single (mocked API)
# ---------------------------------------------------------------------------

class TestVLMValidatorSingle:
    """Tests for validate_single with mocked HTTP calls."""

    def _make_validator(self) -> VLMValidator:
        return VLMValidator(api_key="test-key", veto_threshold=1.5)

    def test_successful_validation_no_veto(self) -> None:
        v = self._make_validator()
        with patch.object(v, "_call_api", return_value=("good", 150.0)):
            result = v.validate_single("img1", "/tmp/img.jpg", "overall", 4.0)
        assert result.vlm_label == "good"
        assert result.vlm_score == pytest.approx(4.0)
        assert result.is_vetoed is False
        assert result.parse_success is True

    def test_successful_validation_with_veto(self) -> None:
        v = self._make_validator()
        # SigLIP2 says 5.0 (excellent), VLM says "poor" (2.0) → disagreement 3.0 >= 1.5
        with patch.object(v, "_call_api", return_value=("poor", 200.0)):
            result = v.validate_single("img2", "/tmp/img.jpg", "overall", 5.0)
        assert result.is_vetoed is True
        assert result.level_disagreement == pytest.approx(3.0)

    def test_api_failure_no_veto(self) -> None:
        v = self._make_validator()
        with patch.object(v, "_call_api", side_effect=RuntimeError("timeout")):
            result = v.validate_single("img3", "/tmp/img.jpg", "overall", 3.0)
        assert result.is_vetoed is False
        assert result.parse_success is False
        assert result.vlm_label is None

    def test_parse_failure_no_veto(self) -> None:
        v = self._make_validator()
        with patch.object(v, "_call_api", return_value=("I cannot tell", 100.0)):
            result = v.validate_single("img4", "/tmp/img.jpg", "overall", 3.0)
        assert result.is_vetoed is False
        assert result.parse_success is False


# ---------------------------------------------------------------------------
# VLMValidator.select_tier2_queue
# ---------------------------------------------------------------------------

class TestSelectTier2Queue:
    """Tests for tier-2 queue selection and capping."""

    def test_caps_at_max_pool_fraction(self) -> None:
        v = VLMValidator(api_key="test-key", max_pool_fraction=0.10)
        candidates = [{"jsd": i * 0.1, "image_id": f"img{i}"} for i in range(20)]
        selected = v.select_tier2_queue(candidates, total_pool_size=100)
        assert len(selected) == 10  # 10% of 100

    def test_sorted_by_descending_jsd(self) -> None:
        v = VLMValidator(api_key="test-key", max_pool_fraction=1.0)
        candidates = [
            {"jsd": 0.1, "image_id": "low"},
            {"jsd": 0.9, "image_id": "high"},
            {"jsd": 0.5, "image_id": "mid"},
        ]
        selected = v.select_tier2_queue(candidates, total_pool_size=3)
        assert [c["image_id"] for c in selected] == ["high", "mid", "low"]

    def test_empty_candidates(self) -> None:
        v = VLMValidator(api_key="test-key")
        selected = v.select_tier2_queue([], total_pool_size=100)
        assert selected == []
