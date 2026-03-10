"""Contract tests for level name/prefix/token consistency.

Verifies that the centralized constants in src/constants.py are consistent
and match the conventions used throughout the codebase.
"""

from __future__ import annotations

import pytest

from src.constants import LEVEL_NAMES, LEVEL_PREFIX, LEVEL_SCORES


class TestLevelConstants:
    """Verify core level constant consistency."""

    def test_level_names_count(self) -> None:
        assert len(LEVEL_NAMES) == 5

    def test_level_scores_count(self) -> None:
        assert len(LEVEL_SCORES) == 5

    def test_names_and_scores_same_length(self) -> None:
        assert len(LEVEL_NAMES) == len(LEVEL_SCORES)

    def test_scores_descending(self) -> None:
        """DeQA convention: excellent=5 → bad=1."""
        for i in range(len(LEVEL_SCORES) - 1):
            assert LEVEL_SCORES[i] > LEVEL_SCORES[i + 1]

    def test_scores_range(self) -> None:
        assert max(LEVEL_SCORES) == 5.0
        assert min(LEVEL_SCORES) == 1.0

    def test_level_names_are_lowercase(self) -> None:
        for name in LEVEL_NAMES:
            assert name == name.lower()

    def test_level_prefix_nonempty(self) -> None:
        assert len(LEVEL_PREFIX) > 0

    def test_level_names_ordering(self) -> None:
        """Canonical ordering: excellent, good, fair, poor, bad."""
        assert LEVEL_NAMES == ["excellent", "good", "fair", "poor", "bad"]


class TestVlmValidatorConsistency:
    """Verify vlm_validator QUALITY_LEVEL_MAP matches constants."""

    def test_quality_level_map_matches_constants(self) -> None:
        from src.uncertainty.vlm_validator import QUALITY_LEVEL_MAP

        assert set(QUALITY_LEVEL_MAP.keys()) == set(LEVEL_NAMES)
        for name, score in zip(LEVEL_NAMES, LEVEL_SCORES):
            assert QUALITY_LEVEL_MAP[name] == score
