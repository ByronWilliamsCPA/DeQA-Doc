"""Tests for the unified metadata schema, I/O, and conversion functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.uncertainty.metadata_schema import (
    SCHEMA_VERSION,
    AcceptanceDecisionRecord,
    AcceptanceTierValue,
    ActiveLearningRecord,
    CrossValidationSignals,
    DimensionRecord,
    DocumentContext,
    HumanLabel,
    ImageMetadataRecord,
    LabelSource,
    OODRecord,
    SigLIP2Prediction,
    UncertaintySignalsRecord,
    VLMEvalRecord,
    VLMVetoRecord,
)
from src.uncertainty.metadata_io import (
    build_index,
    merge_records,
    read_master_jsonl,
    write_master_jsonl,
)
from src.uncertainty.metadata_convert import (
    extract_canonical_id,
    extract_split_from_path,
    from_diqa_test_json,
    from_diqa_training_json,
    from_image_detection_metadata,
    from_pseudo_label_sample,
    from_vlm_eval_jsonl,
    to_training_record,
)


# ── Fixtures ───────────────────────────────────────────────────────────


VALID_LEVEL_PROBS = [0.1, 0.5, 0.3, 0.08, 0.02]


@pytest.fixture
def minimal_record():
    """Minimal valid ImageMetadataRecord (no dimensions, no enrichment)."""
    return ImageMetadataRecord(
        canonical_id="00001",
        dataset="diqa5000",
        image_path_res="DIQA/test/res/test_res_00001.jpg",
    )


@pytest.fixture
def full_record():
    """Fully populated ImageMetadataRecord."""
    return ImageMetadataRecord(
        canonical_id="00042",
        dataset="diqa5000",
        split="train",
        image_path_res="DIQA/train/res/train_res_00042.jpg",
        image_path_ori="DIQA/train/ori/train_ori_00042.jpg",
        image_detection_id="abc123",
        document=DocumentContext(
            domain_level1="EDU",
            iso639_language="zh",
            iso15924_script="Hans",
            capture_method="camera_smartphone",
            resolution_category="standard_300",
            effective_dpi=300,
            has_table=True,
        ),
        ood=OODRecord(
            mahalanobis_distance=28.5,
            is_ood=False,
            threshold=46.0,
            percentile=42.0,
        ),
        vlm_evals=[
            VLMEvalRecord(
                model_id="openai/gpt-4.1",
                overall=4.2,
                sharpness=4.5,
                color_fidelity=4.0,
                reasoning="Good quality document.",
                latency_ms=3000.0,
            ),
        ],
        dimensions={
            "overall": DimensionRecord(
                label_source=LabelSource.HUMAN,
                level_probs=VALID_LEVEL_PROBS,
                mos=3.58,
                std=0.72,
                human=HumanLabel(
                    gt_score=3.58,
                    gt_score_norm=4.0,
                    level_probs=VALID_LEVEL_PROBS,
                    std=0.72,
                    std_norm=0.72,
                ),
            ),
        },
        is_pseudo_labeled=False,
        created_at="2026-03-07T12:00:00+00:00",
        tags={"batch": "initial"},
    )


@pytest.fixture
def sample_training_record():
    """A training JSON record matching existing format."""
    return {
        "id": "image00042.jpg",
        "image": "DIQA/train/res/train_res_00042.jpg",
        "gt_score": 3.187,
        "gt_score_norm": 3.919,
        "level_probs": [0.0, 0.919, 0.081, 0.0, 0.0],
        "conversations": [
            {
                "from": "human",
                "value": "How would you judge the quality of this image?\n<|image|>",
            },
            {"from": "gpt", "value": "The quality of the image is good."},
        ],
        "std": 0.8,
        "std_norm": 0.8,
    }


# ── Schema validation tests ───────────────────────────────────────────


class TestSchemaValidation:
    """Test Pydantic model validation rules."""

    def test_minimal_record_validates(self, minimal_record):
        assert minimal_record.canonical_id == "00001"
        assert minimal_record.schema_version == SCHEMA_VERSION
        assert minimal_record.dimensions == {}
        assert minimal_record.vlm_evals == []

    def test_full_record_validates(self, full_record):
        assert full_record.canonical_id == "00042"
        assert full_record.document.domain_level1 == "EDU"
        assert full_record.ood.mahalanobis_distance == 28.5
        assert len(full_record.vlm_evals) == 1
        assert "overall" in full_record.dimensions

    def test_level_probs_length_validation(self):
        with pytest.raises(Exception):
            DimensionRecord(
                label_source=LabelSource.HUMAN,
                level_probs=[0.5, 0.5, 0.0],  # Wrong length
                mos=3.5,
                std=0.5,
            )

    def test_level_probs_sum_validation(self):
        with pytest.raises(Exception):
            DimensionRecord(
                label_source=LabelSource.HUMAN,
                level_probs=[0.5, 0.5, 0.5, 0.0, 0.0],  # Sums to 1.5
                mos=3.5,
                std=0.5,
            )

    def test_valid_level_probs(self):
        dim = DimensionRecord(
            label_source=LabelSource.HUMAN,
            level_probs=VALID_LEVEL_PROBS,
            mos=3.58,
            std=0.72,
        )
        assert dim.level_probs == VALID_LEVEL_PROBS

    def test_confidence_weight_bounds(self):
        with pytest.raises(Exception):
            AcceptanceDecisionRecord(
                tier=AcceptanceTierValue.AUTO_ACCEPT,
                confidence_weight=1.5,  # > 1.0
                signals=UncertaintySignalsRecord(
                    mahalanobis_distance=20.0,
                    cross_model_jsd=0.03,
                    siglip2_sigma_sq=0.4,
                    siglip2_entropy=0.8,
                ),
                reason="test",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            OODRecord(
                mahalanobis_distance=30.0,
                is_ood=False,
                threshold=46.0,
                unknown_field="bad",
            )

    def test_optional_fields_default_none(self, minimal_record):
        assert minimal_record.split is None
        assert minimal_record.image_path_ori is None
        assert minimal_record.document is None
        assert minimal_record.ood is None
        assert minimal_record.image_detection_id is None
        assert minimal_record.pipeline_run_id is None

    def test_enum_values(self):
        assert LabelSource.HUMAN.value == "human"
        assert AcceptanceTierValue.AUTO_ACCEPT.value == "auto_accept"
        assert LabelSource.PSEUDO_LABEL.value == "pseudo_label"


# ── Round-trip serialization tests ─────────────────────────────────────


class TestRoundTrip:
    """Test JSON serialization round-trips."""

    def test_minimal_roundtrip(self, minimal_record):
        json_str = minimal_record.json()
        restored = ImageMetadataRecord.parse_raw(json_str)
        assert restored.canonical_id == minimal_record.canonical_id
        assert restored.dataset == minimal_record.dataset
        assert restored.schema_version == SCHEMA_VERSION

    def test_full_roundtrip(self, full_record):
        json_str = full_record.json()
        restored = ImageMetadataRecord.parse_raw(json_str)
        assert restored.canonical_id == "00042"
        assert restored.document.domain_level1 == "EDU"
        assert restored.ood.mahalanobis_distance == 28.5
        assert restored.vlm_evals[0].model_id == "openai/gpt-4.1"
        assert restored.dimensions["overall"].label_source == LabelSource.HUMAN
        assert restored.dimensions["overall"].level_probs == VALID_LEVEL_PROBS

    def test_dict_roundtrip(self, full_record):
        data = full_record.dict()
        restored = ImageMetadataRecord(**data)
        assert restored.canonical_id == full_record.canonical_id
        assert restored.document.has_table is True


# ── I/O tests ──────────────────────────────────────────────────────────


class TestIO:
    """Test JSONL read/write operations."""

    def test_write_and_read(self, minimal_record, full_record):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        try:
            count = write_master_jsonl([minimal_record, full_record], path)
            assert count == 2

            records = read_master_jsonl(path)
            assert len(records) == 2
            assert records[0].canonical_id == "00001"
            assert records[1].canonical_id == "00042"
        finally:
            path.unlink(missing_ok=True)

    def test_append_mode(self, minimal_record):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        try:
            write_master_jsonl([minimal_record], path, mode="w")
            write_master_jsonl([minimal_record], path, mode="a")

            records = read_master_jsonl(path)
            assert len(records) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_build_index(self, minimal_record, full_record):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = Path(f.name)

        try:
            write_master_jsonl([minimal_record, full_record], path)
            index = build_index(path)
            assert "00001" in index
            assert "00042" in index
            assert index["00042"].document.domain_level1 == "EDU"
        finally:
            path.unlink(missing_ok=True)

    def test_empty_lines_skipped(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False
        ) as f:
            rec = ImageMetadataRecord(
                canonical_id="00001",
                dataset="test",
                image_path_res="test.jpg",
            )
            f.write(rec.json() + "\n")
            f.write("\n")  # Empty line
            f.write(rec.json() + "\n")
            path = Path(f.name)

        try:
            records = read_master_jsonl(path)
            assert len(records) == 2
        finally:
            path.unlink(missing_ok=True)


# ── Merge tests ────────────────────────────────────────────────────────


class TestMerge:
    """Test record merge logic."""

    def test_merge_different_dimensions(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            split="train",
            image_path_res="DIQA/train/res/train_res_00001.jpg",
            dimensions={
                "overall": DimensionRecord(
                    label_source=LabelSource.HUMAN,
                    level_probs=VALID_LEVEL_PROBS,
                    mos=3.58,
                    std=0.72,
                ),
            },
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="DIQA/train/res/train_res_00001.jpg",
            dimensions={
                "sharpness": DimensionRecord(
                    label_source=LabelSource.HUMAN,
                    level_probs=[0.2, 0.6, 0.15, 0.04, 0.01],
                    mos=3.95,
                    std=0.65,
                ),
            },
        )

        merged = merge_records(rec1, rec2)
        assert "overall" in merged.dimensions
        assert "sharpness" in merged.dimensions
        assert merged.split == "train"

    def test_merge_vlm_evals(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            vlm_evals=[
                VLMEvalRecord(model_id="model_a", overall=4.0),
            ],
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            vlm_evals=[
                VLMEvalRecord(model_id="model_b", overall=3.5),
            ],
        )

        merged = merge_records(rec1, rec2)
        model_ids = {ev.model_id for ev in merged.vlm_evals}
        assert model_ids == {"model_a", "model_b"}

    def test_merge_vlm_dedup(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            vlm_evals=[
                VLMEvalRecord(model_id="model_a", overall=4.0),
            ],
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            vlm_evals=[
                VLMEvalRecord(model_id="model_a", overall=4.5),  # Updated
            ],
        )

        merged = merge_records(rec1, rec2)
        assert len(merged.vlm_evals) == 1
        assert merged.vlm_evals[0].overall == 4.5

    def test_merge_document_context(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            document=DocumentContext(
                domain_level1="SCI",
                iso639_language="en",
                iso15924_script="Latn",
                capture_method="scanner_flatbed",
                resolution_category="standard_300",
            ),
        )

        merged = merge_records(rec1, rec2)
        assert merged.document is not None
        assert merged.document.domain_level1 == "SCI"

    def test_merge_tags(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            tags={"source": "diqa"},
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            tags={"batch": "v2"},
        )

        merged = merge_records(rec1, rec2)
        assert merged.tags == {"source": "diqa", "batch": "v2"}

    def test_merge_mismatched_ids_raises(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="a.jpg",
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00002",
            dataset="diqa5000",
            image_path_res="b.jpg",
        )

        with pytest.raises(ValueError, match="different canonical_ids"):
            merge_records(rec1, rec2)

    def test_merge_updated_at_set(self):
        rec1 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
        )
        rec2 = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
        )

        merged = merge_records(rec1, rec2)
        assert merged.updated_at is not None


# ── Conversion tests ───────────────────────────────────────────────────


class TestCanonicalIdExtraction:
    """Test canonical ID extraction from various filename formats."""

    def test_image_id_format(self):
        assert extract_canonical_id("image00001.jpg") == "00001"

    def test_res_format(self):
        assert extract_canonical_id("test_res_00042.jpg") == "00042"

    def test_ori_format(self):
        assert extract_canonical_id("train_ori_00100.jpg") == "00100"

    def test_no_extension(self):
        assert extract_canonical_id("train_res_00001") == "00001"

    def test_full_path(self):
        assert extract_canonical_id("DIQA/train/res/train_res_00001.jpg") == "00001"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            extract_canonical_id("no_digits.jpg")


class TestSplitExtraction:
    """Test split name extraction from paths."""

    def test_train_path(self):
        assert extract_split_from_path("DIQA/train/res/train_res_00001.jpg") == "train"

    def test_test_filename(self):
        assert extract_split_from_path("test_res_00001.jpg") == "test"

    def test_val_path(self):
        assert extract_split_from_path("DIQA/val/res/val_res_00001.jpg") == "val"

    def test_unknown(self):
        assert extract_split_from_path("image.jpg") is None


class TestFromDiqaTrainingJson:
    """Test conversion from existing training JSON format."""

    def test_basic_conversion(self, sample_training_record):
        records = from_diqa_training_json(
            [sample_training_record], dimension="overall"
        )
        assert len(records) == 1
        rec = records[0]

        assert rec.canonical_id == "00042"
        assert rec.split == "train"
        assert rec.dataset == "diqa5000"
        assert rec.image_path_res == "DIQA/train/res/train_res_00042.jpg"
        assert rec.image_path_ori == "DIQA/train/ori/train_ori_00042.jpg"

        dim = rec.dimensions["overall"]
        assert dim.label_source == LabelSource.HUMAN
        assert dim.human is not None
        assert dim.human.gt_score == 3.187
        assert dim.level_probs == [0.0, 0.919, 0.081, 0.0, 0.0]

    def test_split_override(self, sample_training_record):
        records = from_diqa_training_json(
            [sample_training_record], dimension="overall", split="val"
        )
        assert records[0].split == "val"


class TestFromDiqaTestJson:
    """Test conversion from test manifest (no labels)."""

    def test_basic_conversion(self):
        test_records = [
            {"image": "test_res_00001.jpg"},
            {"image": "test_res_00002.jpg"},
        ]
        records = from_diqa_test_json(test_records)
        assert len(records) == 2

        rec = records[0]
        assert rec.canonical_id == "00001"
        assert rec.split == "test"
        assert rec.image_path_res == "DIQA/test/res/test_res_00001.jpg"
        assert rec.image_path_ori == "DIQA/test/ori/test_ori_00001.jpg"
        assert rec.dimensions == {}


class TestFromVlmEvalJsonl:
    """Test conversion from VLM eval JSONL files."""

    def test_basic_parsing(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False
        ) as f:
            data = {
                "model_id": "openai/gpt-4.1",
                "image": "test_res_00001.jpg",
                "overall": 4.2,
                "sharpness": 4.5,
                "color_fidelity": 4.0,
                "reasoning": "Good quality.",
                "latency_ms": 3000,
                "error": "",
            }
            f.write(json.dumps(data) + "\n")
            path = Path(f.name)

        try:
            result = from_vlm_eval_jsonl(path)
            assert "00001" in result
            assert len(result["00001"]) == 1
            ev = result["00001"][0]
            assert ev.model_id == "openai/gpt-4.1"
            assert ev.overall == 4.2
            assert ev.sharpness == 4.5
        finally:
            path.unlink(missing_ok=True)


class TestFromImageDetectionMetadata:
    """Test conversion from image_detection metadata format."""

    def test_basic_extraction(self):
        sample = {
            "id": "abc123-uuid",
            "enrichments": {
                "versions": [
                    {
                        "version": 1,
                        "data": {
                            "domain_level1": "SCI",
                            "iso639_language": "en",
                            "iso15924_script": "Latn",
                            "capture_method": "scanner_flatbed",
                            "resolution_category": "standard_300",
                            "effective_dpi": 300,
                            "has_table": True,
                            "has_formula": False,
                            "has_handwriting": False,
                            "has_figure": True,
                            "color_mode": "grayscale",
                            "layout_type": "single-column",
                        },
                    }
                ]
            },
        }

        doc_ctx, sample_id = from_image_detection_metadata(sample)
        assert sample_id == "abc123-uuid"
        assert doc_ctx is not None
        assert doc_ctx.domain_level1 == "SCI"
        assert doc_ctx.iso639_language == "en"
        assert doc_ctx.capture_method == "scanner_flatbed"
        assert doc_ctx.has_table is True
        assert doc_ctx.has_figure is True
        assert doc_ctx.color_mode == "grayscale"

    def test_empty_enrichments(self):
        sample = {"id": "empty-uuid", "enrichments": {"versions": []}}
        doc_ctx, sample_id = from_image_detection_metadata(sample)
        assert doc_ctx is None
        assert sample_id == "empty-uuid"

    def test_missing_fields_get_defaults(self):
        sample = {
            "id": "sparse-uuid",
            "enrichments": {
                "versions": [{"version": 1, "data": {}}]
            },
        }
        doc_ctx, _ = from_image_detection_metadata(sample)
        assert doc_ctx is not None
        assert doc_ctx.domain_level1 == "UNK"
        assert doc_ctx.iso639_language == "und"
        assert doc_ctx.has_table is False


class TestToTrainingRecord:
    """Test export to SingleDataset-compatible format."""

    def test_human_label_export(self, full_record):
        rec = to_training_record(full_record, "overall", seed=42)
        assert rec is not None
        assert "image" in rec
        assert rec["image"] == "DIQA/train/res/train_res_00042.jpg"
        assert rec["gt_score"] == 3.58
        assert rec["level_probs"] == [0.1, 0.5, 0.3, 0.08, 0.02]
        assert rec["pseudo_label"] is False
        assert rec["source_tier"] == "human"
        assert len(rec["conversations"]) == 2
        assert rec["conversations"][0]["from"] == "human"
        assert "<|image|>" in rec["conversations"][0]["value"]

    def test_missing_dimension_returns_none(self, full_record):
        rec = to_training_record(full_record, "sharpness")
        assert rec is None

    def test_pseudo_label_export(self):
        metadata = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="DIQA/test/res/test_res_00001.jpg",
            dimensions={
                "overall": DimensionRecord(
                    label_source=LabelSource.PSEUDO_LABEL,
                    level_probs=[0.15, 0.45, 0.30, 0.08, 0.02],
                    mos=3.63,
                    std=0.81,
                    acceptance=AcceptanceDecisionRecord(
                        tier=AcceptanceTierValue.AUTO_ACCEPT,
                        confidence_weight=1.0,
                        signals=UncertaintySignalsRecord(
                            mahalanobis_distance=25.0,
                            cross_model_jsd=0.03,
                            siglip2_sigma_sq=0.4,
                            siglip2_entropy=1.0,
                        ),
                        reason="All signals within auto-accept thresholds",
                    ),
                ),
            },
            is_pseudo_labeled=True,
        )

        rec = to_training_record(metadata, "overall", seed=42)
        assert rec is not None
        assert rec["pseudo_label"] is True
        assert rec["confidence_weight"] == 1.0
        assert rec["source_tier"] == "auto_accept"
        assert rec["id"].startswith("pseudo_")

    def test_level_text_is_argmax(self):
        """The answer text should reflect the argmax of level_probs."""
        metadata = ImageMetadataRecord(
            canonical_id="00001",
            dataset="diqa5000",
            image_path_res="test.jpg",
            dimensions={
                "overall": DimensionRecord(
                    label_source=LabelSource.HUMAN,
                    level_probs=[0.6, 0.2, 0.1, 0.05, 0.05],  # argmax=0=excellent
                    mos=4.25,
                    std=0.9,
                ),
            },
        )
        rec = to_training_record(metadata, "overall", seed=42)
        assert "excellent" in rec["conversations"][1]["value"]


class TestFromPseudoLabelSample:
    """Test conversion from pipeline PseudoLabelSample dataclass."""

    def test_basic_conversion(self):
        """Test with a mock PseudoLabelSample-like object."""
        from types import SimpleNamespace

        signals = SimpleNamespace(
            mahalanobis_distance=30.0,
            cross_model_jsd=0.04,
            siglip2_sigma_sq=0.5,
            siglip2_entropy=1.1,
        )
        decision = SimpleNamespace(
            tier=SimpleNamespace(value="auto_accept"),
            confidence_weight=1.0,
            signals=signals,
            reason="All good",
        )
        sample = SimpleNamespace(
            image_id="DIQA/test/res/test_res_00001.jpg",
            dimension="overall",
            level_probs=np.array([0.1, 0.5, 0.3, 0.08, 0.02]),
            mos=3.58,
            std=0.72,
            confidence_weight=1.0,
            tier=SimpleNamespace(value="auto_accept"),
            decision=decision,
            vlm_vetoed=False,
        )

        metadata = from_pseudo_label_sample(sample)
        assert metadata.canonical_id == "00001"
        assert metadata.is_pseudo_labeled is True
        dim = metadata.dimensions["overall"]
        assert dim.label_source == LabelSource.PSEUDO_LABEL
        assert dim.acceptance is not None
        assert dim.acceptance.tier == AcceptanceTierValue.AUTO_ACCEPT

    def test_with_ood_result(self):
        from types import SimpleNamespace

        sample = SimpleNamespace(
            image_id="test_res_00005.jpg",
            dimension="sharpness",
            level_probs=np.array([0.2, 0.6, 0.15, 0.04, 0.01]),
            mos=3.96,
            std=0.65,
            confidence_weight=0.8,
            tier=SimpleNamespace(value="low_weight"),
            decision=None,
            vlm_vetoed=False,
        )
        ood = SimpleNamespace(
            mahalanobis_distance=38.5,
            is_ood=False,
            threshold=46.0,
            percentile=75.0,
        )

        metadata = from_pseudo_label_sample(sample, ood_result=ood)
        assert metadata.ood is not None
        assert metadata.ood.mahalanobis_distance == 38.5
        assert metadata.ood.percentile == 75.0


# ── Integration test: full pipeline round-trip ─────────────────────────


class TestFullRoundTrip:
    """End-to-end: training JSON → metadata → JSONL → metadata → training JSON."""

    def test_training_json_roundtrip(self, sample_training_record):
        # 1. Convert training JSON to metadata
        metadata_list = from_diqa_training_json(
            [sample_training_record], dimension="overall"
        )
        assert len(metadata_list) == 1

        # 2. Write to JSONL
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", mode="w", delete=False
        ) as f:
            path = Path(f.name)

        try:
            write_master_jsonl(metadata_list, path)

            # 3. Read back from JSONL
            restored = read_master_jsonl(path)
            assert len(restored) == 1

            # 4. Export back to training record
            exported = to_training_record(restored[0], "overall", seed=0)
            assert exported is not None

            # 5. Verify key fields match original
            assert exported["image"] == sample_training_record["image"]
            assert exported["gt_score"] == sample_training_record["gt_score"]
            assert exported["level_probs"] == sample_training_record["level_probs"]
            assert exported["std"] == sample_training_record["std"]
            assert exported["pseudo_label"] is False
        finally:
            path.unlink(missing_ok=True)
