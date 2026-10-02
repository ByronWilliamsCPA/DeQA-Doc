"""Tests for CER, WER, and FCA computation."""

from __future__ import annotations

import pytest

from research.ocr_iqa_correlation.analysis.cer_wer import (
    compute_cer,
    compute_fca,
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


class TestComputeFCA:
    """Tests for Flexible Character Accuracy."""

    def test_identical_strings(self) -> None:
        assert compute_fca("hello world", "hello world") == 0.0

    def test_empty_both(self) -> None:
        assert compute_fca("", "") == 0.0

    def test_empty_reference(self) -> None:
        assert compute_fca("", "some text") == 1.0

    def test_empty_hypothesis(self) -> None:
        assert compute_fca("some text", "") == 1.0

    def test_identical_multiline(self) -> None:
        ref = "line one\nline two\nline three"
        hyp = "line one\nline two\nline three"
        assert compute_fca(ref, hyp) == 0.0

    def test_reordered_lines(self) -> None:
        """FCA should be robust to reading-order changes."""
        ref = "first line\nsecond line\nthird line"
        hyp = "third line\nfirst line\nsecond line"
        # Greedy alignment should find perfect matches for all lines
        assert compute_fca(ref, hyp) == 0.0

    def test_reordered_lines_lower_than_cer(self) -> None:
        """FCA should produce lower error than CER for reordered text."""
        ref = "alpha beta gamma\ndelta epsilon zeta"
        hyp = "delta epsilon zeta\nalpha beta gamma"
        fca = compute_fca(ref, hyp)
        cer = compute_cer(ref, hyp)
        assert fca < cer, f"FCA ({fca}) should be less than CER ({cer}) for reordered lines"

    def test_partial_match_multiline(self) -> None:
        ref = "hello world\ngoodbye moon"
        hyp = "hello world\ngoodbye mars"
        fca = compute_fca(ref, hyp)
        assert 0.0 < fca < 1.0

    def test_extra_hyp_lines_penalized(self) -> None:
        """Extra hypothesis lines should increase FCA."""
        ref = "hello world"
        hyp = "hello world\nextra line one\nextra line two"
        fca = compute_fca(ref, hyp)
        assert fca > 0.0, "Extra hypothesis lines should be penalized"

    def test_missing_hyp_lines_penalized(self) -> None:
        """Missing hypothesis lines should increase FCA."""
        ref = "line one\nline two\nline three"
        hyp = "line one"
        fca = compute_fca(ref, hyp)
        assert fca > 0.0, "Missing hypothesis lines should be penalized"

    def test_single_line_matches_cer(self) -> None:
        """With single-line input, FCA should equal CER."""
        ref = "hello world"
        hyp = "hallo world"
        fca = compute_fca(ref, hyp)
        cer = compute_cer(ref, hyp)
        assert fca == pytest.approx(cer, abs=0.01)

    def test_case_insensitive(self) -> None:
        """FCA should be 0 when only case differs."""
        assert compute_fca("Hello World", "hello world") == 0.0

    def test_blank_lines_ignored(self) -> None:
        """Blank lines should not affect FCA."""
        ref = "hello\n\nworld"
        hyp = "hello\nworld"
        assert compute_fca(ref, hyp) == 0.0


class TestComputeMetrics:
    """Tests for combined metrics."""

    def test_returns_both(self) -> None:
        result = compute_metrics("hello world", "hello world")
        assert "cer" in result
        assert "wer" in result
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0

    def test_returns_fca_when_requested(self) -> None:
        result = compute_metrics("hello world", "hello world", include_fca=True)
        assert "fca" in result
        assert result["fca"] == 0.0

    def test_no_fca_by_default(self) -> None:
        result = compute_metrics("hello world", "hello world")
        assert "fca" not in result
