"""Conversion functions between existing data formats and metadata schema.

Bridges:
    1. DIQA training JSON → ImageMetadataRecord (human labels)
    2. PseudoLabelSample → ImageMetadataRecord (pipeline output)
    3. VLM eval JSONL → VLMEvalRecord list (per-model checkpoints)
    4. image_detection metadata → DocumentContext (snapshot)
    5. ImageMetadataRecord → training JSON dict (SingleDataset export)
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .format_training_data import ANSWER_TEMPLATES, LEVEL_NAMES, QUESTIONS
from .metadata_schema import (
    SCHEMA_VERSION,
    AcceptanceDecisionRecord,
    AcceptanceTierValue,
    DimensionRecord,
    DocumentContext,
    HumanLabel,
    ImageMetadataRecord,
    LabelSource,
    OODRecord,
    UncertaintySignalsRecord,
    VLMEvalRecord,
    VLMVetoRecord,
)


def extract_canonical_id(filename: str) -> str:
    """Extract the numeric canonical ID from a DIQA filename.

    Examples:
        "image00001.jpg" → "00001"
        "test_res_00001.jpg" → "00001"
        "train_ori_00001.jpg" → "00001"
        "train_res_00001" → "00001"

    Args:
        filename: Filename or path string.

    Returns:
        Zero-padded numeric ID string (e.g., "00001").

    Raises:
        ValueError: If no numeric ID can be extracted.
    """
    # Strip path and extension
    basename = Path(filename).stem
    # Find the last sequence of digits
    match = re.search(r"(\d{3,})$", basename)
    if match:
        return match.group(1)
    # Fallback: find any digit sequence
    match = re.search(r"(\d{3,})", basename)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract canonical ID from: {filename}")


def extract_split_from_path(image_path: str) -> Optional[str]:
    """Extract split name from an image path.

    Examples:
        "DIQA/train/res/train_res_00001.jpg" → "train"
        "DIQA/test/res/test_res_00001.jpg" → "test"
        "test_res_00001.jpg" → "test"
        "val_res_00001.jpg" → "val"

    Args:
        image_path: Image path or filename.

    Returns:
        Split name or None if not determinable.
    """
    path_lower = image_path.lower()
    for split in ("train", "test", "val"):
        if split in path_lower:
            return split
    return None


# ── 1. DIQA training JSON → ImageMetadataRecord ───────────────────────


def from_diqa_training_json(
    records: List[dict],
    dimension: str,
    split: Optional[str] = None,
) -> List[ImageMetadataRecord]:
    """Convert existing DIQA training JSON records to metadata records.

    Each input record is per-dimension (e.g., from train_diqa_overall.json).
    Creates an ImageMetadataRecord with the human label populated for the
    given dimension.

    Args:
        records: List of dicts from the training JSON file.
        dimension: Quality dimension ("overall", "sharpness", "color").
        split: Override split name. If None, extracted from image path.

    Returns:
        List of ImageMetadataRecord instances.
    """
    results = []
    now = datetime.now(timezone.utc).isoformat()

    for rec in records:
        image_path = rec.get("image", "")
        rec_id = rec.get("id", image_path)

        canonical_id = extract_canonical_id(rec_id)
        detected_split = split or extract_split_from_path(image_path)

        # Build human label
        level_probs = rec.get("level_probs", [0.0, 0.0, 0.0, 0.0, 0.0])
        gt_score = rec.get("gt_score", 0.0)
        gt_score_norm = rec.get("gt_score_norm")
        std = rec.get("std", 0.0)
        std_norm = rec.get("std_norm")

        human_label = HumanLabel(
            gt_score=gt_score,
            gt_score_norm=gt_score_norm,
            level_probs=level_probs,
            std=std,
            std_norm=std_norm,
        )

        dim_record = DimensionRecord(
            label_source=LabelSource.HUMAN,
            level_probs=level_probs,
            mos=gt_score,
            std=std,
            human=human_label,
        )

        # Build ori path from res path
        ori_path = image_path.replace("/res/", "/ori/").replace("_res_", "_ori_")

        metadata = ImageMetadataRecord(
            schema_version=SCHEMA_VERSION,
            canonical_id=canonical_id,
            dataset="diqa5000",
            split=detected_split,
            image_path_res=image_path,
            image_path_ori=ori_path if ori_path != image_path else None,
            dimensions={dimension: dim_record},
            created_at=now,
        )
        results.append(metadata)

    return results


def from_diqa_test_json(records: List[dict]) -> List[ImageMetadataRecord]:
    """Convert DIQA test manifest (no labels) to minimal metadata records.

    Args:
        records: List of dicts from diqa_test.json (only 'image' field).

    Returns:
        List of ImageMetadataRecord instances with no dimension data.
    """
    results = []
    now = datetime.now(timezone.utc).isoformat()

    for rec in records:
        image_filename = rec.get("image", "")
        canonical_id = extract_canonical_id(image_filename)

        # Build full paths
        res_path = f"DIQA/test/res/{image_filename}"
        ori_filename = image_filename.replace("_res_", "_ori_")
        ori_path = f"DIQA/test/ori/{ori_filename}"

        metadata = ImageMetadataRecord(
            schema_version=SCHEMA_VERSION,
            canonical_id=canonical_id,
            dataset="diqa5000",
            split="test",
            image_path_res=res_path,
            image_path_ori=ori_path,
            created_at=now,
        )
        results.append(metadata)

    return results


# ── 2. PseudoLabelSample → ImageMetadataRecord ────────────────────────


def from_pseudo_label_sample(
    sample: Any,
    ood_result: Optional[Any] = None,
) -> ImageMetadataRecord:
    """Convert a PseudoLabelSample to an ImageMetadataRecord.

    Args:
        sample: PseudoLabelSample from the pipeline.
        ood_result: Optional OODResult from ood_wrapper.

    Returns:
        ImageMetadataRecord with the pseudo-label dimension populated.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Convert level_probs from numpy
    level_probs = [round(float(p), 6) for p in sample.level_probs]

    # Build acceptance decision record if available
    acceptance = None
    if sample.decision is not None:
        signals = UncertaintySignalsRecord(
            mahalanobis_distance=sample.decision.signals.mahalanobis_distance,
            cross_model_jsd=sample.decision.signals.cross_model_jsd,
            siglip2_sigma_sq=sample.decision.signals.siglip2_sigma_sq,
            siglip2_entropy=sample.decision.signals.siglip2_entropy,
        )
        acceptance = AcceptanceDecisionRecord(
            tier=AcceptanceTierValue(sample.decision.tier.value),
            confidence_weight=sample.decision.confidence_weight,
            signals=signals,
            reason=sample.decision.reason,
        )

    # Build VLM veto if present
    vlm_veto = None
    if sample.vlm_vetoed:
        vlm_veto = VLMVetoRecord(is_vetoed=True, vlm_model="unknown")

    dim_record = DimensionRecord(
        label_source=LabelSource.PSEUDO_LABEL,
        level_probs=level_probs,
        mos=round(float(sample.mos), 4),
        std=round(float(sample.std), 4),
        acceptance=acceptance,
        vlm_veto=vlm_veto,
    )

    # Build OOD record
    ood = None
    if ood_result is not None:
        ood = OODRecord(
            mahalanobis_distance=ood_result.mahalanobis_distance,
            is_ood=ood_result.is_ood,
            threshold=ood_result.threshold,
            percentile=getattr(ood_result, "percentile", None),
        )

    canonical_id = extract_canonical_id(sample.image_id)

    return ImageMetadataRecord(
        schema_version=SCHEMA_VERSION,
        canonical_id=canonical_id,
        dataset="diqa5000",
        split=extract_split_from_path(sample.image_id),
        image_path_res=sample.image_id,
        ood=ood,
        dimensions={sample.dimension: dim_record},
        is_pseudo_labeled=True,
        created_at=now,
    )


# ── 3. VLM eval JSONL → VLMEvalRecord ─────────────────────────────────


def from_vlm_eval_jsonl(
    path: Union[str, Path],
) -> Dict[str, List[VLMEvalRecord]]:
    """Read a per-model VLM eval JSONL file and return records by image ID.

    Args:
        path: Path to VLM eval JSONL (e.g., openai__gpt-4.1.jsonl).

    Returns:
        Dict mapping canonical_id to list of VLMEvalRecord.
    """
    path = Path(path)
    result: Dict[str, List[VLMEvalRecord]] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            image_field = data.get("image", "")
            canonical_id = extract_canonical_id(image_field)

            record = VLMEvalRecord(
                model_id=data.get("model_id", ""),
                overall=data.get("overall"),
                sharpness=data.get("sharpness"),
                color_fidelity=data.get("color_fidelity"),
                reasoning=data.get("reasoning", ""),
                latency_ms=data.get("latency_ms", 0.0),
                error=data.get("error", ""),
            )

            result.setdefault(canonical_id, []).append(record)

    return result


# ── 4. image_detection metadata → DocumentContext ──────────────────────


def from_image_detection_metadata(
    sample_dict: dict,
) -> Tuple[Optional[DocumentContext], Optional[str]]:
    """Extract DocumentContext snapshot from an image_detection sample.

    Reads the latest enrichment version's data fields and returns a
    DocumentContext with quality-relevant fields plus the sample UUID.

    Args:
        sample_dict: A single sample dict from diqa-5000_metadata.json.

    Returns:
        Tuple of (DocumentContext or None, image_detection_id or None).
    """
    sample_id = sample_dict.get("id")

    enrichments = sample_dict.get("enrichments", {})
    versions = enrichments.get("versions", [])
    if not versions:
        return None, sample_id

    # Use the latest version's data
    latest_data = versions[-1].get("data", {})

    # Extract snapshot fields (all optional in source)
    domain = latest_data.get("domain_level1", "UNK")
    language = latest_data.get("iso639_language", "und")
    script = latest_data.get("iso15924_script", "Zyyy")
    capture = latest_data.get("capture_method", "unknown")
    resolution_cat = latest_data.get("resolution_category", "unknown")

    doc_context = DocumentContext(
        domain_level1=domain,
        iso639_language=language,
        iso15924_script=script,
        capture_method=capture,
        resolution_category=resolution_cat,
        effective_dpi=latest_data.get("effective_dpi"),
        orientation_class=latest_data.get("orientation_class"),
        skew_angle_degrees=latest_data.get("skew_angle_degrees"),
        color_mode=latest_data.get("color_mode"),
        layout_type=latest_data.get("layout_type"),
        has_table=latest_data.get("has_table", False),
        has_formula=latest_data.get("has_formula", False),
        has_handwriting=latest_data.get("has_handwriting", False),
        has_figure=latest_data.get("has_figure", False),
    )

    return doc_context, sample_id


# ── 5. ImageMetadataRecord → training JSON dict ───────────────────────


def to_training_record(
    metadata: ImageMetadataRecord,
    dimension: str,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """Export a metadata record to SingleDataset-compatible training JSON.

    Output format matches ``format_training_data.sample_to_training_record``.

    Args:
        metadata: The source metadata record.
        dimension: Quality dimension to export ("overall", "sharpness", "color").
        seed: Random seed for question template selection.

    Returns:
        Dict ready for JSON serialization, or None if dimension not present.
    """
    dim_rec = metadata.dimensions.get(dimension)
    if dim_rec is None:
        return None

    rng = random.Random(seed) if seed is not None else random

    level_text = LEVEL_NAMES[int(np.argmax(dim_rec.level_probs))]
    answer_template = ANSWER_TEMPLATES.get(dimension, ANSWER_TEMPLATES["overall"])
    question = rng.choice(QUESTIONS)
    answer = answer_template.format(level_text)

    # Determine confidence weight and tier
    confidence_weight = 1.0
    source_tier = "human"
    if dim_rec.acceptance is not None:
        confidence_weight = dim_rec.acceptance.confidence_weight
        source_tier = dim_rec.acceptance.tier.value
    elif dim_rec.label_source == LabelSource.HUMAN:
        confidence_weight = 1.0
        source_tier = "human"

    is_pseudo = dim_rec.label_source != LabelSource.HUMAN

    return {
        "id": f"{'pseudo_' if is_pseudo else ''}{dimension}_{metadata.canonical_id}",
        "image": metadata.image_path_res,
        "gt_score": round(dim_rec.mos, 4),
        "gt_score_norm": round(dim_rec.mos, 4),
        "level_probs": [round(float(p), 6) for p in dim_rec.level_probs],
        "conversations": [
            {"from": "human", "value": f"{question}\n<|image|>"},
            {"from": "gpt", "value": answer},
        ],
        "std": round(dim_rec.std, 4),
        "std_norm": round(dim_rec.std, 4),
        # Extra metadata (ignored by SingleDataset)
        "pseudo_label": is_pseudo,
        "confidence_weight": round(confidence_weight, 4),
        "source_tier": source_tier,
    }
