"""Tests for format_training_data.py."""

import json

import numpy as np

from src.uncertainty.format_training_data import (
    sample_to_training_record,
    samples_to_training_json,
)
from src.uncertainty.fusion import (
    AcceptanceDecision,
    AcceptanceTier,
    UncertaintySignals,
)
from src.uncertainty.pseudo_label import PseudoLabelSample


def _make_sample(
    image_id: str = "test_image.jpg",
    dimension: str = "overall",
    level_probs: np.ndarray | None = None,
    confidence_weight: float = 1.0,
    tier: AcceptanceTier = AcceptanceTier.AUTO_ACCEPT,
) -> PseudoLabelSample:
    """Create a test PseudoLabelSample."""
    if level_probs is None:
        level_probs = np.array([0.1, 0.3, 0.4, 0.15, 0.05])

    signals = UncertaintySignals(
        mahalanobis_distance=25.0,
        cross_model_jsd=0.03,
        siglip2_sigma_sq=0.3,
        siglip2_entropy=0.8,
    )
    decision = AcceptanceDecision(
        image_id=image_id,
        dimension=dimension,
        tier=tier,
        confidence_weight=confidence_weight,
        signals=signals,
        reason="test",
    )

    return PseudoLabelSample(
        image_id=image_id,
        dimension=dimension,
        level_probs=level_probs,
        mos=3.2,
        std=0.8,
        confidence_weight=confidence_weight,
        tier=tier,
        decision=decision,
    )


class TestSampleToTrainingRecord:
    def test_required_fields(self):
        record = sample_to_training_record(_make_sample(), seed=42)
        required = [
            "id",
            "image",
            "gt_score",
            "gt_score_norm",
            "level_probs",
            "conversations",
            "std",
            "std_norm",
        ]
        for field in required:
            assert field in record, f"Missing required field: {field}"

    def test_level_probs_length(self):
        record = sample_to_training_record(_make_sample(), seed=42)
        assert len(record["level_probs"]) == 5

    def test_level_probs_sums_to_one(self):
        record = sample_to_training_record(_make_sample(), seed=42)
        assert abs(sum(record["level_probs"]) - 1.0) < 1e-4

    def test_conversations_format(self):
        record = sample_to_training_record(_make_sample(), seed=42)
        convs = record["conversations"]
        assert len(convs) == 2
        assert convs[0]["from"] == "human"
        assert convs[1]["from"] == "gpt"
        assert "<|image|>" in convs[0]["value"]

    def test_answer_contains_level(self):
        probs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # All mass on bad
        record = sample_to_training_record(_make_sample(level_probs=probs), seed=42)
        assert "bad" in record["conversations"][1]["value"]

    def test_metadata_fields(self):
        record = sample_to_training_record(_make_sample(), seed=42)
        assert record["pseudo_label"] is True
        assert "confidence_weight" in record
        assert "source_tier" in record

    def test_dimension_specific_answer(self):
        record = sample_to_training_record(_make_sample(dimension="sharpness"), seed=42)
        assert "sharpness" in record["conversations"][1]["value"]

    def test_image_prefix(self):
        record = sample_to_training_record(
            _make_sample(image_id="img001.jpg"),
            image_prefix="DIQA/unlabeled",
            seed=42,
        )
        assert record["image"].startswith("DIQA/unlabeled/")


class TestSamplesToTrainingJson:
    def test_writes_valid_json(self, tmp_path):
        samples = [_make_sample(image_id=f"img{i:03d}.jpg") for i in range(5)]
        output = tmp_path / "pseudo.json"
        count = samples_to_training_json(samples, output, seed=42)
        assert count == 5

        with open(output) as f:
            data = json.load(f)
        assert len(data) == 5

    def test_filters_by_weight(self, tmp_path):
        samples = [
            _make_sample(image_id="high.jpg", confidence_weight=1.0),
            _make_sample(image_id="low.jpg", confidence_weight=0.1),
        ]
        output = tmp_path / "pseudo.json"
        count = samples_to_training_json(samples, output, min_weight=0.3, seed=42)
        assert count == 1

    def test_filters_vetoed(self, tmp_path):
        sample = _make_sample(image_id="vetoed.jpg")
        sample.vlm_vetoed = True
        output = tmp_path / "pseudo.json"
        count = samples_to_training_json([sample], output, seed=42)
        assert count == 0

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "sub" / "dir" / "pseudo.json"
        samples_to_training_json([_make_sample()], output, seed=42)
        assert output.exists()
