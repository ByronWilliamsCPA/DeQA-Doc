"""Tests for FUNSD annotation parser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research.ocr_iqa_correlation.gt_extraction.funsd_parser import (
    parse_funsd_annotation,
)


@pytest.fixture
def sample_annotation(tmp_path: Path) -> Path:
    """Create a sample FUNSD annotation file."""
    annotation = {
        "form": [
            {
                "id": 0,
                "text": "R&D",
                "label": "other",
                "words": [{"text": "R&D", "box": [292, 91, 376, 175]}],
                "linking": [],
                "box": [292, 91, 376, 175],
            },
            {
                "id": 1,
                "text": "DEPARTMENT:",
                "label": "question",
                "words": [{"text": "DEPARTMENT:", "box": [100, 200, 250, 220]}],
                "linking": [[1, 2]],
                "box": [100, 200, 250, 220],
            },
            {
                "id": 2,
                "text": "Engineering",
                "label": "answer",
                "words": [{"text": "Engineering", "box": [260, 200, 380, 220]}],
                "linking": [],
                "box": [260, 200, 380, 220],
            },
        ]
    }
    ann_path = tmp_path / "test_annotation.json"
    ann_path.write_text(json.dumps(annotation))
    return ann_path


def test_parse_annotation_concatenates_text(sample_annotation: Path) -> None:
    """Entities are joined in id order with spaces."""
    text = parse_funsd_annotation(sample_annotation)
    assert text == "R&D DEPARTMENT: Engineering"


def test_parse_annotation_sorts_by_id(tmp_path: Path) -> None:
    """Entities are sorted by id regardless of list order."""
    annotation = {
        "form": [
            {"id": 2, "text": "third", "words": [], "linking": [], "box": [0, 0, 1, 1]},
            {"id": 0, "text": "first", "words": [], "linking": [], "box": [0, 0, 1, 1]},
            {"id": 1, "text": "second", "words": [], "linking": [], "box": [0, 0, 1, 1]},
        ]
    }
    ann_path = tmp_path / "test.json"
    ann_path.write_text(json.dumps(annotation))

    text = parse_funsd_annotation(ann_path)
    assert text == "first second third"


def test_parse_annotation_skips_empty_text(tmp_path: Path) -> None:
    """Empty text entries are filtered out."""
    annotation = {
        "form": [
            {"id": 0, "text": "hello", "words": [], "linking": [], "box": [0, 0, 1, 1]},
            {"id": 1, "text": "", "words": [], "linking": [], "box": [0, 0, 1, 1]},
            {"id": 2, "text": "  ", "words": [], "linking": [], "box": [0, 0, 1, 1]},
            {"id": 3, "text": "world", "words": [], "linking": [], "box": [0, 0, 1, 1]},
        ]
    }
    ann_path = tmp_path / "test.json"
    ann_path.write_text(json.dumps(annotation))

    text = parse_funsd_annotation(ann_path)
    assert text == "hello world"
