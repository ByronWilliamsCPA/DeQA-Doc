"""Configuration for OCR-IQA correlation experiment.

Centralizes paths, tier definitions, seeds, and experiment parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────

# Source datasets on E: drive
IMAGE_DETECTION_BASE = Path("/mnt/e/image_detection")
FUNSD_ROOT = IMAGE_DETECTION_BASE / "01_base_data" / "forms" / "funsd"
FUNSD_PLUS_ROOT = IMAGE_DETECTION_BASE / "01_base_data" / "forms" / "funsd_plus"

# FUNSD split paths
FUNSD_TRAIN_IMAGES = FUNSD_ROOT / "train" / "images"
FUNSD_TRAIN_ANNOTATIONS = FUNSD_ROOT / "train" / "annotations"
FUNSD_TEST_IMAGES = FUNSD_ROOT / "test" / "images"
FUNSD_TEST_ANNOTATIONS = FUNSD_ROOT / "test" / "annotations"

# FUNSD+ split paths
FUNSD_PLUS_IMAGES = FUNSD_PLUS_ROOT / "images"
FUNSD_PLUS_TRAIN_ARROW = FUNSD_PLUS_ROOT / "train" / "data-00000-of-00001.arrow"
FUNSD_PLUS_TEST_ARROW = FUNSD_PLUS_ROOT / "test" / "data-00000-of-00001.arrow"

# image_detection source for distortion pipeline
IMAGE_DETECTION_SRC = Path("/home/byron/dev/image_detection/src")

# Project output paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GT_TEXT_DIR = DATA_DIR / "gt_text"
BASE_IMAGES_DIR = DATA_DIR / "base_images"
DISTORTED_DIR = DATA_DIR / "distorted"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
DEQA_RESULTS_DIR = DATA_DIR / "deqa_results"
DATASET_JSONL = DATA_DIR / "dataset.jsonl"
SAMPLE_MANIFEST = DATA_DIR / "sample_manifest.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"


# ── Quality Tiers ──────────────────────────────────────────────────────────


class QualityTier(str, Enum):
    """Quality tier for distortion levels.

    Maps to HybridProfile from image_detection's augmentation pipeline.
    """

    ORIGINAL = "ORIGINAL"
    PRISTINE = "PRISTINE"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DEGRADED = "DEGRADED"


# Maps our tier names to HybridProfile enum values
TIER_TO_HYBRID_PROFILE = {
    QualityTier.PRISTINE: "pristine",
    QualityTier.HIGH: "light",
    QualityTier.MEDIUM: "moderate",
    QualityTier.LOW: "heavy",
    QualityTier.DEGRADED: "historical",
}

# Expected overall_quality ranges per tier
TIER_QUALITY_RANGES: dict[QualityTier, tuple[float, float]] = {
    QualityTier.ORIGINAL: (0.95, 1.0),
    QualityTier.PRISTINE: (0.95, 1.0),
    QualityTier.HIGH: (0.70, 0.95),
    QualityTier.MEDIUM: (0.40, 0.80),
    QualityTier.LOW: (0.10, 0.60),
    QualityTier.DEGRADED: (0.00, 0.50),
}

DISTORTION_TIERS = [
    QualityTier.PRISTINE,
    QualityTier.HIGH,
    QualityTier.MEDIUM,
    QualityTier.LOW,
    QualityTier.DEGRADED,
]


# ── Experiment Parameters ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration parameters."""

    # Sampling
    total_base_images: int = 200
    funsd_sample_count: int = 50
    funsd_plus_sample_count: int = 150
    min_gt_chars: int = 20
    random_seed: int = 42

    # Distortion
    base_distortion_seed: int = 10000
    augraphy_probability: float = 0.7
    albumentations_probability: float = 1.0

    # OCR engines
    ocr_engines: list[str] = field(
        default_factory=lambda: [
            "tesseract",
            "rapidocr",
            "easyocr",
            "gcloud_vision",
        ]
    )

    # DeQA
    deqa_batch_size: int = 8


DEFAULT_CONFIG = ExperimentConfig()
