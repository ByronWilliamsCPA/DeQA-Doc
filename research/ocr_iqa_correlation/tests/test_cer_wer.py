"""Tests for CER and WER computation."""

from __future__ import annotations

import pytest

from research.ocr_iqa_correlation.analysis.cer_wer import (
    compute_cer,
    compute_metrics,
    compute_wer,
    normalize_text,
)


class TestNormalizeText:
    """Tests for text normalization."""

    def test_lowercase(self) -> None:
        assert normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        assert normalize_text("hello   world") == "hello world"

    def test_strip(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_unicode_normalization(self) -> None:
        # NFC normalization: combining characters → precomposed
        assert normalize_text("café") == normalize_text("café")

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""


class TestComputeCER:
    """Tests for Character Error Rate."""

    def test_identical_strings(self) -> None:
        assert compute_cer("hello world", "hello world") == 0.0

    def test_empty_both(self) -> None:
        assert compute_cer("", "") == 0.0

    def test_empty_reference(self) -> None:
        assert compute_cer("", "some text") == 1.0

    def test_empty_hypothesis(self) -> None:
        assert compute_cer("some text", "") == 1.0

    def test_partial_match(self) -> None:
        cer = compute_cer("hello", "hallo")
        assert 0.0 < cer < 1.0

    def test_case_insensitive(self) -> None:
        """CER should be 0 when only case differs."""
        assert compute_cer("Hello World", "hello world") == 0.0


class TestComputeWER:
    """Tests for Word Error Rate."""

    def test_identical_strings(self) -> None:
        assert compute_wer("hello world", "hello world") == 0.0

    def test_empty_both(self) -> None:
        assert compute_wer("", "") == 0.0

    def test_one_word_wrong(self) -> None:
        wer = compute_wer("hello world", "hello earth")
        assert wer == pytest.approx(0.5, abs=0.01)

    def test_all_words_wrong(self) -> None:
        wer = compute_wer("hello world", "goodbye earth")
        assert wer == pytest.approx(1.0, abs=0.01)


class TestComputeMetrics:
    """Tests for combined metrics."""

    def test_returns_both(self) -> None:
        result = compute_metrics("hello world", "hello world")
        assert "cer" in result
        assert "wer" in result
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
