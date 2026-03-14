"""Stream 3: Two-model VLM consensus labeling for real document datasets.

Uses the VLM ensemble (Gemini 3.1 Flash Lite + Qwen 3.5-122B) as a label
source for real document images that lack human quality annotations.
Training weight = 0.5 (lower than deterministic due to label noise).

Target sources and counts:
    OHR-Bench:    1,200 (text GT available → OCR cross-validation)
    RealDAE:        600 (text GT available → OCR cross-validation)
    Tobacco800:     400 (no text GT)
    SmartDoc-QA:    500 (text GT available → OCR cross-validation)
    Total:        2,700 samples, ~$5.40

VLM consensus protocol:
    1. Both models agree (within 1 MOS) → use mean as label
    2. Models disagree (>1 MOS) → invoke tiebreaker (GPT-4.1)
    3. Parse failure → retry once, then skip
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .iqa_to_deqa import vlm_scores_to_deqa_record

logger = logging.getLogger(__name__)

# VLM label weight (lower than deterministic due to noise)
VLM_WEIGHT = 0.5

# VLM model IDs
PRIMARY_MODEL = "google/gemini-3.1-flash-lite-preview"
SECONDARY_MODEL = "qwen/qwen3.5-122b-a10b"
TIEBREAKER_MODEL = "openai/gpt-4.1"


@dataclass
class SourceDataset:
    """Configuration for a real document dataset to label via VLM."""

    name: str
    image_dir: str | Path
    count: int  # target samples to label
    has_text_gt: bool = False  # whether OCR cross-validation is possible
    text_gt_dir: str | Path | None = None
    glob_pattern: str = "*.png"
    description: str = ""


# Tier 1 source datasets
TIER1_SOURCES = [
    SourceDataset(
        name="ohr_bench",
        image_dir="",  # To be configured at runtime
        count=1200,
        has_text_gt=True,
        description="OHR-Bench: real degraded documents with OCR ground truth",
    ),
    SourceDataset(
        name="realdae",
        image_dir="",
        count=600,
        has_text_gt=True,
        description="RealDAE: real document augmentation evaluation set",
    ),
    SourceDataset(
        name="tobacco800",
        image_dir="",
        count=400,
        has_text_gt=False,
        description="Tobacco800: scanned legacy documents",
    ),
    SourceDataset(
        name="smartdoc_qa",
        image_dir="",
        count=500,
        has_text_gt=True,
        description="SmartDoc-QA: camera-captured documents with text GT",
    ),
]


@dataclass
class VLMConsensusResult:
    """Result of two-model VLM consensus for a single image."""

    image_id: str
    image_path: str
    primary_label: str | None  # "excellent", "good", etc.
    primary_score: float | None  # MOS from primary
    secondary_label: str | None
    secondary_score: float | None
    tiebreaker_label: str | None = None
    tiebreaker_score: float | None = None
    consensus_mos: float | None = None
    consensus_std: float = 0.8
    agreement: bool = False  # True if primary and secondary agree within 1 MOS
    tiebreaker_used: bool = False
    parse_success: bool = True


@dataclass
class ConsensusTracker:
    """Tracks VLM consensus labeling progress and costs."""

    total_images: int = 0
    labeled: int = 0
    agreements: int = 0
    tiebreakers: int = 0
    failures: int = 0
    cost_usd: float = 0.0
    results_by_source: dict[str, int] = field(default_factory=dict)

    # Estimated costs per call (based on ~1500 input + 5 output tokens)
    PRIMARY_COST: float = 0.0  # Flash Lite is free preview
    SECONDARY_COST: float = 0.00027 + 0.0000025  # $0.18/1M in + $0.50/1M out
    TIEBREAKER_COST: float = 0.003 + 0.00004  # $2/1M in + $8/1M out

    def record_result(
        self, result: VLMConsensusResult, source: str
    ) -> None:
        """Record a consensus result."""
        self.total_images += 1
        if result.consensus_mos is not None:
            self.labeled += 1
            self.results_by_source[source] = (
                self.results_by_source.get(source, 0) + 1
            )
            # Cost: always 2 calls (primary + secondary)
            self.cost_usd += self.PRIMARY_COST + self.SECONDARY_COST
            if result.agreement:
                self.agreements += 1
            if result.tiebreaker_used:
                self.tiebreakers += 1
                self.cost_usd += self.TIEBREAKER_COST
        else:
            self.failures += 1

    def summary(self) -> dict:
        """Return summary stats."""
        return {
            "total_images": self.total_images,
            "labeled": self.labeled,
            "agreements": self.agreements,
            "tiebreakers": self.tiebreakers,
            "failures": self.failures,
            "cost_usd": round(self.cost_usd, 4),
            "agreement_rate": (
                self.agreements / self.labeled if self.labeled > 0 else 0.0
            ),
            "results_by_source": dict(self.results_by_source),
        }


def compute_consensus(
    primary_score: float | None,
    secondary_score: float | None,
    tiebreaker_score: float | None = None,
    disagreement_threshold: float = 1.0,
) -> tuple[float | None, float, bool, bool]:
    """Compute consensus MOS from multi-model scores.

    Args:
        primary_score: MOS from primary model.
        secondary_score: MOS from secondary model.
        tiebreaker_score: MOS from tiebreaker (if invoked).
        disagreement_threshold: MOS difference that triggers tiebreaker.

    Returns:
        Tuple of (consensus_mos, consensus_std, agreement, tiebreaker_used).
    """
    if primary_score is None and secondary_score is None:
        return None, 0.8, False, False

    if primary_score is None:
        return secondary_score, 0.8, False, False
    if secondary_score is None:
        return primary_score, 0.8, False, False

    disagreement = abs(primary_score - secondary_score)
    agreement = disagreement <= disagreement_threshold

    if agreement:
        # Simple mean when models agree
        consensus = (primary_score + secondary_score) / 2.0
        std = disagreement / 2.0 + 0.4  # base uncertainty + half disagreement
        return consensus, std, True, False

    # Models disagree — use tiebreaker if available
    if tiebreaker_score is not None:
        # Median of three
        scores = sorted([primary_score, secondary_score, tiebreaker_score])
        consensus = scores[1]
        import numpy as np

        std = float(np.std([primary_score, secondary_score, tiebreaker_score]))
        return consensus, max(std, 0.4), False, True

    # No tiebreaker: use mean but flag higher uncertainty
    consensus = (primary_score + secondary_score) / 2.0
    std = disagreement / 2.0 + 0.4
    return consensus, std, False, False


def discover_images(
    source: SourceDataset,
    seed: int = 42,
) -> list[Path]:
    """Discover and sample images from a source dataset.

    Args:
        source: Source dataset configuration.
        seed: Random seed for sampling.

    Returns:
        List of image paths, up to source.count.
    """
    import random

    image_dir = Path(source.image_dir)
    if not image_dir.exists():
        logger.warning("Source directory not found: %s", image_dir)
        return []

    images = sorted(image_dir.glob(source.glob_pattern))
    # Also check for jpg
    if not images:
        images = sorted(image_dir.glob("*.jpg"))
    if not images:
        images = sorted(image_dir.glob("*.jpeg"))

    rng = random.Random(seed)
    if len(images) > source.count:
        rng.shuffle(images)
        images = images[: source.count]

    logger.info(
        "Source %s: %d images available, selected %d",
        source.name, len(list(image_dir.glob("*"))), len(images),
    )
    return images


def label_single_image(
    image_path: str | Path,
    validator,  # VLMValidator instance
    dimension: str = "overall",
) -> VLMConsensusResult:
    """Label a single image using VLM consensus.

    Uses the existing VLMValidator's _call_api method for API calls,
    but computes standalone consensus (not veto against SigLIP2).

    Args:
        image_path: Path to image file.
        validator: VLMValidator instance (from src.uncertainty.vlm_validator).
        dimension: Quality dimension to assess.

    Returns:
        VLMConsensusResult with consensus MOS.
    """
    from src.uncertainty.vlm_validator import (
        QUALITY_LEVEL_MAP,
        _parse_vlm_response,
    )

    image_path = str(image_path)
    image_id = Path(image_path).stem

    primary_model = validator._models_by_role.get("primary")
    secondary_model = validator._models_by_role.get("secondary")
    tiebreaker_model = validator._models_by_role.get("tiebreaker")

    # Call primary
    primary_label = None
    primary_score = None
    try:
        text, _ = validator._call_api(image_path, dimension, primary_model)
        primary_label = _parse_vlm_response(text)
        if primary_label:
            primary_score = QUALITY_LEVEL_MAP[primary_label]
        validator.budget.record_call(
            vetoed=False, parse_success=primary_label is not None,
            model_config=primary_model,
        )
    except Exception:
        logger.warning("Primary model failed for %s", image_id)

    # Call secondary
    secondary_label = None
    secondary_score = None
    try:
        text, _ = validator._call_api(image_path, dimension, secondary_model)
        secondary_label = _parse_vlm_response(text)
        if secondary_label:
            secondary_score = QUALITY_LEVEL_MAP[secondary_label]
        validator.budget.record_call(
            vetoed=False, parse_success=secondary_label is not None,
            model_config=secondary_model,
        )
    except Exception:
        logger.warning("Secondary model failed for %s", image_id)

    # Check if tiebreaker needed
    tiebreaker_label = None
    tiebreaker_score = None
    tiebreaker_used = False

    if (
        primary_score is not None
        and secondary_score is not None
        and abs(primary_score - secondary_score) > validator.tiebreaker_threshold
    ):
        try:
            text, _ = validator._call_api(image_path, dimension, tiebreaker_model)
            tiebreaker_label = _parse_vlm_response(text)
            if tiebreaker_label:
                tiebreaker_score = QUALITY_LEVEL_MAP[tiebreaker_label]
            tiebreaker_used = True
            validator.budget.record_call(
                vetoed=False, parse_success=tiebreaker_label is not None,
                model_config=tiebreaker_model,
            )
            validator.budget.tiebreaker_invocations += 1
        except Exception:
            logger.warning("Tiebreaker failed for %s", image_id)

    # Compute consensus
    consensus_mos, consensus_std, agreement, _ = compute_consensus(
        primary_score, secondary_score, tiebreaker_score,
        disagreement_threshold=validator.tiebreaker_threshold,
    )

    parse_success = primary_label is not None or secondary_label is not None

    return VLMConsensusResult(
        image_id=image_id,
        image_path=str(image_path),
        primary_label=primary_label,
        primary_score=primary_score,
        secondary_label=secondary_label,
        secondary_score=secondary_score,
        tiebreaker_label=tiebreaker_label,
        tiebreaker_score=tiebreaker_score,
        consensus_mos=consensus_mos,
        consensus_std=consensus_std,
        agreement=agreement,
        tiebreaker_used=tiebreaker_used,
        parse_success=parse_success,
    )


def generate_stream3(
    sources: list[SourceDataset] | None = None,
    output_dir: str | Path = "",
    api_key: str | None = None,
    dimension: str = "overall",
    base_seed: int = 40000,
    dry_run: bool = False,
    save_vlm_results: bool = True,
) -> list[dict]:
    """Generate Stream 3 VLM consensus-labeled samples.

    Full pipeline:
    1. Discover images from configured source datasets
    2. Label each image with two-model VLM consensus
    3. Convert consensus labels to DeQA training records

    Args:
        sources: List of source datasets. Default: TIER1_SOURCES.
        output_dir: Directory for DIQA-5000_1 output.
        api_key: OpenRouter API key.
        dimension: Quality dimension to assess.
        base_seed: Base seed for reproducibility.
        dry_run: If True, plan only without API calls.
        save_vlm_results: Save raw VLM results alongside training records.

    Returns:
        List of DeQA training records (dicts).
    """
    from src.uncertainty.vlm_validator import VLMValidator

    sources = sources or TIER1_SOURCES
    output_dir = Path(output_dir)

    # Validate sources have image_dirs configured
    active_sources = [s for s in sources if s.image_dir]
    if not active_sources:
        logger.error(
            "No source datasets have image_dir configured. "
            "Set image_dir on each SourceDataset before calling generate_stream3."
        )
        return []

    if dry_run:
        total = sum(s.count for s in active_sources)
        logger.info("Dry run plan: %d images from %d sources", total, len(active_sources))
        for src in active_sources:
            logger.info(
                "  %s: %d images (text_gt=%s)", src.name, src.count, src.has_text_gt
            )
        cost_2model = total * 0.002
        logger.info("  Estimated cost: $%.2f (2-model consensus)", cost_2model)
        return []

    # Initialize VLM validator
    validator = VLMValidator(api_key=api_key)
    tracker = ConsensusTracker()

    records = []
    vlm_results = []

    for source in active_sources:
        images = discover_images(source, seed=base_seed)
        logger.info("Processing source %s: %d images", source.name, len(images))

        for idx, image_path in enumerate(images):
            result = label_single_image(image_path, validator, dimension=dimension)
            tracker.record_result(result, source.name)

            if result.consensus_mos is None:
                continue

            # Copy image to output
            images_out = output_dir / "images" / f"stream3_{source.name}"
            images_out.mkdir(parents=True, exist_ok=True)
            out_filename = f"{source.name}_{result.image_id}.png"

            # Symlink to avoid copying large files
            out_path = images_out / out_filename
            if not out_path.exists():
                import shutil
                shutil.copy2(image_path, out_path)

            # Convert to DeQA record
            rel_path = f"DIQA-5000_1/images/stream3_{source.name}/{out_filename}"
            record = vlm_scores_to_deqa_record(
                image_id=f"s3_{source.name}_{result.image_id}",
                image_path=rel_path,
                vlm_mos=result.consensus_mos,
                vlm_std=result.consensus_std,
                source=source.name,
                weight=VLM_WEIGHT,
                dimension=dimension,
                seed=base_seed + idx,
                vlm_models=[PRIMARY_MODEL, SECONDARY_MODEL],
            )
            record["has_text_gt"] = source.has_text_gt
            record["vlm_agreement"] = result.agreement
            record["tiebreaker_used"] = result.tiebreaker_used
            records.append(record)

            # Save raw VLM result
            vlm_results.append({
                "image_id": result.image_id,
                "source": source.name,
                "primary": {"label": result.primary_label, "score": result.primary_score},
                "secondary": {"label": result.secondary_label, "score": result.secondary_score},
                "tiebreaker": {
                    "label": result.tiebreaker_label,
                    "score": result.tiebreaker_score,
                    "used": result.tiebreaker_used,
                },
                "consensus_mos": result.consensus_mos,
                "consensus_std": result.consensus_std,
                "agreement": result.agreement,
            })

            if (idx + 1) % 100 == 0:
                logger.info(
                    "  %s progress: %d/%d (cost: $%.4f)",
                    source.name, idx + 1, len(images), tracker.cost_usd,
                )

    # Save VLM results
    if save_vlm_results and vlm_results:
        vlm_out = output_dir / "vlm_consensus_results.json"
        vlm_out.parent.mkdir(parents=True, exist_ok=True)
        with open(vlm_out, "w") as f:
            json.dump(vlm_results, f, indent=2)
        logger.info("Saved %d VLM results to %s", len(vlm_results), vlm_out)

    # Log summary
    summary = tracker.summary()
    logger.info("Stream 3 complete: %s", json.dumps(summary, indent=2))

    return records
