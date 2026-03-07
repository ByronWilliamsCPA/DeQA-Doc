"""Convert pseudo-label pipeline output to DeQA training JSON format.

Outputs JSON files consumable by SingleDataset without any modifications
to the training code. Extra metadata fields (pseudo_label, confidence_weight,
source_tier) are ignored by SingleDataset's .get() calls.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .gaussian_to_discrete import level_probs_to_mos, level_probs_to_std
from .pseudo_label import PseudoLabelSample

# Question templates from gen_soft_label.py
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

# Answer templates per dimension
ANSWER_TEMPLATES: dict[str, str] = {
    "overall": "The quality of the image is {}.",
    "sharpness": "The sharpness of the image is {}.",
    "color": "The color_fidelity of the image is {}.",
}

# Level names in DeQA convention (index 0 = highest quality)
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]


def _get_level_text(probs: np.ndarray) -> str:
    """Get the text label for the argmax quality level."""
    return LEVEL_NAMES[int(np.argmax(probs))]


def sample_to_training_record(
    sample: PseudoLabelSample,
    image_prefix: str = "",
    seed: int | None = None,
) -> dict:
    """Convert a PseudoLabelSample to a training JSON record.

    Output format matches gen_soft_label.py / SingleDataset expectations:
        {
            "id": str,
            "image": str,
            "gt_score": float,
            "gt_score_norm": float,
            "level_probs": [float, float, float, float, float],
            "conversations": [...],
            "std": float,
            "std_norm": float,
            "pseudo_label": true,
            "confidence_weight": float,
            "source_tier": str
        }

    Args:
        sample: PseudoLabelSample from the pipeline.
        image_prefix: Optional prefix to prepend to image path.
        seed: Random seed for question selection (None for random).

    Returns:
        Dict ready for JSON serialization.
    """
    rng = random.Random(seed) if seed is not None else random

    mos = level_probs_to_mos(sample.level_probs)
    std = level_probs_to_std(sample.level_probs)
    level_text = _get_level_text(sample.level_probs)

    answer_template = ANSWER_TEMPLATES.get(
        sample.dimension, ANSWER_TEMPLATES["overall"]
    )
    question = rng.choice(QUESTIONS)
    answer = answer_template.format(level_text)

    image_path = sample.image_id
    if image_prefix and not image_path.startswith(image_prefix):
        image_path = f"{image_prefix}/{image_path}"

    return {
        "id": f"pseudo_{sample.dimension}_{sample.image_id}",
        "image": image_path,
        "gt_score": round(mos, 4),
        "gt_score_norm": round(mos, 4),
        "level_probs": [round(float(p), 6) for p in sample.level_probs],
        "conversations": [
            {"from": "human", "value": f"{question}\n<|image|>"},
            {"from": "gpt", "value": answer},
        ],
        "std": round(std, 4),
        "std_norm": round(std, 4),
        # Extra metadata (ignored by SingleDataset)
        "pseudo_label": True,
        "confidence_weight": round(sample.confidence_weight, 4),
        "source_tier": sample.tier.value,
    }


def samples_to_training_json(
    samples: list[PseudoLabelSample],
    output_path: str | Path,
    image_prefix: str = "",
    min_weight: float = 0.3,
    seed: int = 42,
) -> int:
    """Convert a batch of samples to DeQA training JSON file.

    Args:
        samples: List of PseudoLabelSamples.
        output_path: Path for output JSON file.
        image_prefix: Optional prefix for image paths.
        min_weight: Minimum confidence weight to include.
        seed: Random seed for reproducible question selection.

    Returns:
        Number of samples written.
    """
    records = []
    for i, sample in enumerate(samples):
        if sample.confidence_weight < min_weight:
            continue
        if sample.vlm_vetoed:
            continue
        record = sample_to_training_record(
            sample, image_prefix=image_prefix, seed=seed + i
        )
        records.append(record)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    return len(records)


def generate_per_dimension_json(
    samples: list[PseudoLabelSample],
    output_dir: str | Path,
    image_prefix: str = "",
    min_weight: float = 0.3,
    seed: int = 42,
) -> dict[str, int]:
    """Generate separate training JSON files per dimension.

    Creates files like:
        output_dir/pseudo_overall.json
        output_dir/pseudo_sharpness.json
        output_dir/pseudo_color.json

    Args:
        samples: List of PseudoLabelSamples.
        output_dir: Directory for output files.
        image_prefix: Optional prefix for image paths.
        min_weight: Minimum confidence weight to include.
        seed: Random seed.

    Returns:
        Dict mapping dimension to number of samples written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dimension: dict[str, list[PseudoLabelSample]] = {}
    for sample in samples:
        by_dimension.setdefault(sample.dimension, []).append(sample)

    counts = {}
    for dim, dim_samples in by_dimension.items():
        output_path = output_dir / f"pseudo_{dim}.json"
        counts[dim] = samples_to_training_json(
            dim_samples,
            output_path=output_path,
            image_prefix=image_prefix,
            min_weight=min_weight,
            seed=seed,
        )

    return counts
