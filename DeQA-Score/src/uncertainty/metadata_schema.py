"""Unified metadata schema for the DIQA training pipeline.

Provides Pydantic models for per-image metadata records that track every
signal produced across the pseudo-labeling pipeline: human labels, OOD
detection, SigLIP2 predictions, cross-validation, VLM teacher evaluations,
acceptance decisions, and active learning scores.

The same schema serves both DIQA-5000 images (with human labels) and new
unlabeled images that receive pseudo-labels. All sub-schemas are optional
to support incremental enrichment.

Uses Pydantic v1 API (matching the pinned ``pydantic<2`` constraint).

Level ordering convention (CRITICAL):
    Index 0 = excellent (score 5) → Index 4 = bad (score 1)
    MOS = dot(level_probs, [5, 4, 3, 2, 1])
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

SCHEMA_VERSION = "1.0.0"

# DeQA level ordering: index 0 = excellent (best), index 4 = bad (worst)
LEVEL_NAMES = ("excellent", "good", "fair", "poor", "bad")
MOS_WEIGHTS = (5.0, 4.0, 3.0, 2.0, 1.0)


# ── Enums ──────────────────────────────────────────────────────────────


class LabelSource(str, Enum):
    """Provenance of a quality label."""

    HUMAN = "human"
    SIGLIP2 = "siglip2"
    DEQA_SPECIALIST = "deqa_specialist"
    DEQA_ENSEMBLE = "deqa_ensemble"
    VLM_CONSENSUS = "vlm_consensus"
    PSEUDO_LABEL = "pseudo_label"


class AcceptanceTierValue(str, Enum):
    """Tiered acceptance decisions from uncertainty fusion."""

    AUTO_ACCEPT = "auto_accept"
    LOW_WEIGHT = "low_weight"
    TIER2_TRIGGER = "tier2_trigger"
    HARD_REJECT = "hard_reject"


# ── Document context (snapshot from image_detection project) ───────────


class DocumentContext(BaseModel):
    """Quality-relevant document metadata imported from image_detection.

    Snapshot of fields that directly affect quality assessment or enable
    stratified analysis. Full metadata available via ``image_detection_id``.
    """

    domain_level1: str = Field(
        ..., description="3-letter domain code: EDU, SCI, TEC, ADM, etc."
    )
    iso639_language: str = Field(
        ..., description="ISO 639-1/3 language code, e.g. 'zh', 'en'"
    )
    iso15924_script: str = Field(
        ..., description="ISO 15924 script code, e.g. 'Hans', 'Latn'"
    )
    capture_method: str = Field(
        ...,
        description="Capture method: scanner_flatbed, camera_smartphone, etc.",
    )
    resolution_category: str = Field(..., description="low, medium, standard_300, high")
    effective_dpi: Optional[int] = None
    orientation_class: Optional[int] = Field(
        None, description="Detected orientation: 0, 90, 180, or 270"
    )
    skew_angle_degrees: Optional[float] = None
    color_mode: Optional[str] = Field(
        None, description="color, grayscale, or binarized"
    )
    layout_type: Optional[str] = Field(
        None, description="single-column, multi-column, complex, etc."
    )
    has_table: bool = False
    has_formula: bool = False
    has_handwriting: bool = False
    has_figure: bool = False

    class Config:
        extra = "forbid"


# ── Per-image signals ──────────────────────────────────────────────────


class OODRecord(BaseModel):
    """Mahalanobis OOD detection result (per-image, shared across dims)."""

    mahalanobis_distance: float
    is_ood: bool
    threshold: float
    percentile: Optional[float] = None

    class Config:
        extra = "forbid"


class VLMEvalRecord(BaseModel):
    """Single VLM teacher evaluation for one image."""

    model_id: str
    overall: Optional[float] = None
    sharpness: Optional[float] = None
    color_fidelity: Optional[float] = None
    reasoning: str = ""
    latency_ms: float = 0.0
    error: str = ""

    class Config:
        extra = "forbid"


# ── Per-dimension sub-schemas ──────────────────────────────────────────


def _validate_level_probs(v: List[float]) -> List[float]:
    """Validate level_probs is length 5 and approximately sums to 1."""
    if len(v) != 5:
        raise ValueError(f"level_probs must have exactly 5 elements, got {len(v)}")
    total = sum(v)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"level_probs must sum to ~1.0, got {total:.4f}")
    return v


class HumanLabel(BaseModel):
    """Ground-truth label from human annotators (DIQA-5000 only)."""

    gt_score: float = Field(..., ge=0.0, le=5.0)
    gt_score_norm: Optional[float] = None
    level_probs: List[float]
    std: float = Field(..., ge=0.0)
    std_norm: Optional[float] = None

    _validate_probs = validator("level_probs", allow_reuse=True)(_validate_level_probs)

    class Config:
        extra = "forbid"


class SigLIP2Prediction(BaseModel):
    """SigLIP2 regression head output for one dimension."""

    mu: float
    sigma_sq: float = Field(..., ge=0.0)
    level_probs: List[float]
    mos: float
    std: float = Field(..., ge=0.0)

    _validate_probs = validator("level_probs", allow_reuse=True)(_validate_level_probs)

    class Config:
        extra = "forbid"


class CrossValidationSignals(BaseModel):
    """Cross-validation between SigLIP2 and DeQA for one dimension."""

    deqa_source: LabelSource
    deqa_probs: List[float]
    deqa_mos: float
    cross_model_jsd: float = Field(..., ge=0.0)
    mos_delta: float
    siglip2_entropy: float = Field(..., ge=0.0)

    _validate_probs = validator("deqa_probs", allow_reuse=True)(_validate_level_probs)

    class Config:
        extra = "forbid"


class SpreadRecord(BaseModel):
    """Multi-model spread result for one image.

    Records the inter-model disagreement signal computed by
    ``SpreadComputer``. Shared across dimensions (spread is per-image,
    not per-dimension).
    """

    spread: float = Field(..., ge=0.0)
    cluster_divergence: float = Field(..., ge=0.0)
    ood_category: int = Field(
        ..., ge=0, le=2, description="0=in-dist, 1=soft OOD, 2=strong OOD"
    )
    n_models: int = Field(..., ge=0)

    class Config:
        extra = "forbid"


class UncertaintySignalsRecord(BaseModel):
    """Five-signal uncertainty values for one dimension.

    Mirrors ``UncertaintySignals`` dataclass from ``fusion.py``.
    """

    mahalanobis_distance: float
    cross_model_jsd: float
    siglip2_sigma_sq: float
    siglip2_entropy: float
    # Signal 5: Multi-model spread (optional for backward compat)
    model_spread: float = 0.0
    cluster_divergence: float = 0.0
    n_spread_models: int = 0

    class Config:
        extra = "forbid"


class AcceptanceDecisionRecord(BaseModel):
    """Pipeline acceptance decision for one dimension.

    Mirrors ``AcceptanceDecision`` dataclass from ``fusion.py``.
    """

    tier: AcceptanceTierValue
    confidence_weight: float = Field(..., ge=0.0, le=1.0)
    signals: UncertaintySignalsRecord
    reason: str

    class Config:
        extra = "forbid"


class VLMVetoRecord(BaseModel):
    """Tier-2 VLM veto result for one dimension."""

    is_vetoed: bool
    vlm_model: str
    vlm_label: Optional[str] = None
    vlm_score: Optional[float] = None
    level_disagreement: float = 0.0
    latency_ms: float = 0.0

    class Config:
        extra = "forbid"


class ActiveLearningRecord(BaseModel):
    """Active learning score for one dimension."""

    bald: float = Field(..., ge=0.0)
    entropy: float = Field(..., ge=0.0)
    iteration: Optional[int] = None

    class Config:
        extra = "forbid"


class DimensionRecord(BaseModel):
    """All data for a single quality dimension of one image.

    The top-level ``label_source``, ``level_probs``, ``mos``, and ``std``
    represent the **final** label used for training. The nested sub-schemas
    preserve the full provenance chain.
    """

    label_source: LabelSource
    level_probs: List[float]
    mos: float
    std: float = Field(..., ge=0.0)

    # Provenance chain (all optional, populated incrementally)
    human: Optional[HumanLabel] = None
    siglip2: Optional[SigLIP2Prediction] = None
    cross_validation: Optional[CrossValidationSignals] = None
    acceptance: Optional[AcceptanceDecisionRecord] = None
    vlm_veto: Optional[VLMVetoRecord] = None
    active_learning: Optional[ActiveLearningRecord] = None

    _validate_probs = validator("level_probs", allow_reuse=True)(_validate_level_probs)

    class Config:
        extra = "forbid"


# ── Top-level per-image record ─────────────────────────────────────────


class ImageMetadataRecord(BaseModel):
    """Master metadata record for one image across all pipeline layers.

    This is the per-line schema for the master JSONL file. All optional
    fields support incremental enrichment, a minimal record needs only
    ``canonical_id``, ``dataset``, and ``image_path_res``.
    """

    schema_version: str = SCHEMA_VERSION
    canonical_id: str = Field(..., description="Numeric ID portion, e.g. '00001'")
    dataset: str = Field(..., description="Dataset identifier, e.g. 'diqa5000'")
    split: Optional[str] = Field(
        None, description="train, val, test, or None for unlabeled"
    )
    image_path_res: str = Field(
        ..., description="Relative path to resized image from data root"
    )
    image_path_ori: Optional[str] = Field(
        None, description="Relative path to original image"
    )

    # Cross-project reference
    image_detection_id: Optional[str] = Field(
        None, description="UUID from image_detection metadata"
    )

    # Document context (snapshot from image_detection)
    document: Optional[DocumentContext] = None

    # Per-image signals
    ood: Optional[OODRecord] = None
    spread: Optional[SpreadRecord] = None
    vlm_evals: List[VLMEvalRecord] = Field(default_factory=list)

    # Per-dimension quality data
    dimensions: Dict[str, DimensionRecord] = Field(default_factory=dict)

    # Record metadata
    is_pseudo_labeled: bool = False
    pipeline_run_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    class Config:
        extra = "forbid"
