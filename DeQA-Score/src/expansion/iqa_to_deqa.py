"""Convert IQA quality scores to DeQA training format.

Bridges the gap between the image_detection IQA framework (8-dimensional,
overall_quality in [0,1]) and DeQA's training format (MOS in [1,5],
5-bin soft-label distribution [excellent, good, fair, poor, bad]).

Conversion chain:
    overall_quality [0,1] → MOS [1,5] → soft-label [5-bin] → training JSON
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.uncertainty.gaussian_to_discrete import (
    binary_level_probs,
    gaussian_to_level_probs,
    level_probs_to_mos,
    level_probs_to_std,
)


# Question templates (from gen_soft_label.py / format_training_data.py)
QUESTIONS = [
    "What do you think about the quality of this image?",
    "Can you rate the quality of this picture?",
    "Can you judge the quality of this image?",
    "How would you rate the quality of this image?",
    "How would you judge the quality of this image?",
    "What is your quality rating for this image?",
    "What's your opinion on the quality of this picture?",
    "Rate the quality of this image.",
    "Could you evaluate the quality of this image?",
    "How do you assess the quality of this image?",
]

LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]


@dataclass(frozen=True)
class IQALabels:
    """8-dimensional IQA label set from the augmentation pipeline.

    All values in [0, 1] where 0 = no degradation, 1 = maximum degradation
    EXCEPT overall_quality where 1.0 = perfect quality, 0.0 = worst quality.
    """

    blur: float = 0.0
    noise: float = 0.0
    compression: float = 0.0
    ink_degradation: float = 0.0
    paper_degradation: float = 0.0
    geometric_distortion: float = 0.0
    bleed_through: float = 0.0
    overall_quality: float = 1.0


def overall_quality_to_mos(overall_quality: float) -> float:
    """Convert IQA overall_quality [0,1] to DeQA MOS [1,5].

    Linear mapping: MOS = 1.0 + 4.0 * overall_quality
    This maps:
        overall_quality = 0.0 → MOS = 1.0 (bad)
        overall_quality = 0.5 → MOS = 3.0 (fair)
        overall_quality = 1.0 → MOS = 5.0 (excellent)

    Args:
        overall_quality: Quality score in [0, 1].

    Returns:
        MOS score in [1.0, 5.0].
    """
    clamped = max(0.0, min(1.0, overall_quality))
    return 1.0 + 4.0 * clamped


def mos_to_level_probs(
    mos: float,
    sigma: float = 0.8,
    binary_threshold: float = 0.15,
) -> np.ndarray:
    """Convert MOS to 5-bin soft-label distribution.

    Uses Gaussian CDF integration for smooth distributions, or binary
    interpolation when sigma is below threshold.

    Args:
        mos: Mean Opinion Score in [1.0, 5.0].
        sigma: Label uncertainty. Default 0.8 matches DeQA's σ_pseudo.
        binary_threshold: Below this σ, use binary interpolation.

    Returns:
        Array of shape (5,) in DeQA convention [excellent→bad].
    """
    if sigma < binary_threshold:
        return binary_level_probs(mos)
    return gaussian_to_level_probs(mos, sigma)


def iqa_to_deqa_record(
    image_id: str,
    image_path: str,
    iqa_labels: IQALabels | dict,
    *,
    source: str = "",
    stream: str = "",
    sigma: float = 0.8,
    weight: float = 0.7,
    dimension: str = "overall",
    seed: int | None = None,
) -> dict:
    """Convert IQA labels to a DeQA training JSON record.

    Produces a record compatible with SingleDataset's .get() method.

    Args:
        image_id: Unique identifier for the image.
        image_path: Relative path to image file (from Data-DeQA-Score/).
        iqa_labels: IQA quality labels (dataclass or dict).
        source: Source dataset name (e.g., "diqa_degradation", "synth_multiscript").
        stream: Expansion stream (e.g., "stream1_degradation").
        sigma: Label uncertainty for soft-label generation.
        weight: Confidence/training weight for this sample.
        dimension: Quality dimension ("overall", "sharpness", "color").
        seed: Random seed for question selection.

    Returns:
        Dict ready for JSON serialization, compatible with SingleDataset.
    """
    rng = random.Random(seed) if seed is not None else random

    # Extract overall_quality
    if isinstance(iqa_labels, dict):
        overall_quality = iqa_labels.get("overall_quality", 1.0)
    else:
        overall_quality = iqa_labels.overall_quality

    # Convert chain: overall_quality → MOS → level_probs
    mos = overall_quality_to_mos(overall_quality)
    level_probs = mos_to_level_probs(mos, sigma=sigma)

    # Reconstruct MOS from level_probs for consistency
    mos_reconstructed = level_probs_to_mos(level_probs)
    std = level_probs_to_std(level_probs)

    # Pick level text from argmax
    level_text = LEVEL_NAMES[int(np.argmax(level_probs))]

    # Build conversation
    question = rng.choice(QUESTIONS)
    answer_templates = {
        "overall": "The quality of the image is {}.",
        "sharpness": "The sharpness of the image is {}.",
        "color": "The color_fidelity of the image is {}.",
    }
    answer = answer_templates.get(dimension, answer_templates["overall"]).format(
        level_text
    )

    return {
        "id": image_id,
        "image": image_path,
        "gt_score": round(mos_reconstructed, 4),
        "gt_score_norm": round(mos_reconstructed, 4),
        "level_probs": [round(float(p), 6) for p in level_probs],
        "conversations": [
            {"from": "human", "value": f"{question}\n<|image|>"},
            {"from": "gpt", "value": answer},
        ],
        "std": round(std, 4),
        "std_norm": round(std, 4),
        # Expansion metadata (ignored by SingleDataset)
        "pseudo_label": True,
        "confidence_weight": round(weight, 4),
        "source": source,
        "stream": stream,
    }


def vlm_scores_to_deqa_record(
    image_id: str,
    image_path: str,
    vlm_mos: float,
    *,
    vlm_std: float = 0.8,
    source: str = "",
    weight: float = 0.5,
    dimension: str = "overall",
    seed: int | None = None,
    vlm_models: list[str] | None = None,
) -> dict:
    """Convert VLM consensus MOS to a DeQA training record.

    Similar to iqa_to_deqa_record but takes MOS directly from VLM consensus
    rather than computing from IQA labels.

    Args:
        image_id: Unique identifier for the image.
        image_path: Relative path to image file.
        vlm_mos: Consensus MOS from VLM ensemble.
        vlm_std: Standard deviation of VLM predictions.
        source: Source dataset name.
        weight: Training weight (default 0.5 for VLM labels).
        dimension: Quality dimension.
        seed: Random seed for question selection.
        vlm_models: List of VLM model names used for labeling.

    Returns:
        Dict ready for JSON serialization.
    """
    rng = random.Random(seed) if seed is not None else random

    mos = float(np.clip(vlm_mos, 1.0, 5.0))
    level_probs = mos_to_level_probs(mos, sigma=vlm_std)
    mos_reconstructed = level_probs_to_mos(level_probs)
    std = level_probs_to_std(level_probs)

    level_text = LEVEL_NAMES[int(np.argmax(level_probs))]
    question = rng.choice(QUESTIONS)
    answer_templates = {
        "overall": "The quality of the image is {}.",
        "sharpness": "The sharpness of the image is {}.",
        "color": "The color_fidelity of the image is {}.",
    }
    answer = answer_templates.get(dimension, answer_templates["overall"]).format(
        level_text
    )

    record = {
        "id": image_id,
        "image": image_path,
        "gt_score": round(mos_reconstructed, 4),
        "gt_score_norm": round(mos_reconstructed, 4),
        "level_probs": [round(float(p), 6) for p in level_probs],
        "conversations": [
            {"from": "human", "value": f"{question}\n<|image|>"},
            {"from": "gpt", "value": answer},
        ],
        "std": round(std, 4),
        "std_norm": round(std, 4),
        # VLM metadata
        "pseudo_label": True,
        "confidence_weight": round(weight, 4),
        "source": source,
        "stream": "stream3_vlm",
        "vlm_mos_raw": round(vlm_mos, 4),
        "vlm_std_raw": round(vlm_std, 4),
    }
    if vlm_models:
        record["vlm_models"] = vlm_models

    return record
