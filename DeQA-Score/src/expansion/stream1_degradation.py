"""Stream 1: Controlled degradation of DIQA-5000 base images.

Applies 4 degradation levels (HIGH, MEDIUM, LOW, DEGRADED) to selected
DIQA-5000 base images using Albumentations transforms. Labels are
deterministic (computed from degradation parameters), weight = 0.7.

Target: 350 base images × 4 levels = 1,400 new samples.

Degradation profiles are self-contained (no external project dependency).
They replicate the same effect categories as HybridAugmentationPipeline:
blur, noise, compression, geometric distortion, and paper degradation.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .iqa_to_deqa import IQALabels, iqa_to_deqa_record

logger = logging.getLogger(__name__)

# Training weight for deterministic labels
DETERMINISTIC_WEIGHT = 0.7


@dataclass(frozen=True)
class DegradationProfile:
    """Parameters for a single degradation tier."""

    name: str
    tier_label: str
    # Severity ranges [min, max] for each effect
    blur_limit: tuple[int, int]  # kernel size for GaussianBlur
    noise_var: tuple[float, float]  # Gaussian noise variance range
    jpeg_quality: tuple[int, int]  # JPEG quality range
    brightness_range: tuple[float, float]  # brightness shift
    contrast_range: tuple[float, float]  # contrast multiplier
    rotation_limit: int  # max rotation degrees
    # Expected overall_quality range
    quality_range: tuple[float, float]


# 4 degradation tiers (skip PRISTINE — too close to original)
DEGRADATION_PROFILES = [
    DegradationProfile(
        name="light", tier_label="HIGH",
        blur_limit=(3, 5), noise_var=(0.001, 0.005),
        jpeg_quality=(70, 90), brightness_range=(-0.05, 0.05),
        contrast_range=(0.9, 1.1), rotation_limit=2,
        quality_range=(0.70, 0.95),
    ),
    DegradationProfile(
        name="moderate", tier_label="MEDIUM",
        blur_limit=(5, 9), noise_var=(0.005, 0.02),
        jpeg_quality=(40, 70), brightness_range=(-0.1, 0.1),
        contrast_range=(0.7, 1.0), rotation_limit=5,
        quality_range=(0.40, 0.80),
    ),
    DegradationProfile(
        name="heavy", tier_label="LOW",
        blur_limit=(7, 15), noise_var=(0.02, 0.06),
        jpeg_quality=(15, 40), brightness_range=(-0.2, 0.15),
        contrast_range=(0.5, 0.85), rotation_limit=8,
        quality_range=(0.10, 0.60),
    ),
    DegradationProfile(
        name="historical", tier_label="DEGRADED",
        blur_limit=(9, 21), noise_var=(0.04, 0.10),
        jpeg_quality=(5, 20), brightness_range=(-0.3, 0.1),
        contrast_range=(0.3, 0.7), rotation_limit=10,
        quality_range=(0.00, 0.50),
    ),
]

TIER_NAMES = [p.tier_label for p in DEGRADATION_PROFILES]


def select_base_images(
    train_json_path: str | Path,
    num_bases: int = 350,
    strategy: str = "stride",
    seed: int = 42,
) -> list[dict]:
    """Select base images from DIQA-5000 training set for degradation.

    The DIQA-5000 training set has 3,500 samples (350 base × 10 variants).
    This function selects representative base images to degrade further.

    Args:
        train_json_path: Path to train_diqa_overall.json.
        num_bases: Number of base images to select.
        strategy: Selection strategy:
            "stride" — every Nth image (evenly spaced)
            "highest_quality" — prioritize highest MOS (most room to degrade)
            "stratified" — sample proportionally from each quality bin
        seed: Random seed for non-deterministic strategies.

    Returns:
        List of dicts with keys: image_id, image_path, gt_score, original_index.
    """
    with open(train_json_path) as f:
        train_data = json.load(f)

    total = len(train_data)
    if num_bases > total:
        num_bases = total

    if strategy == "stride":
        stride = total // num_bases
        indices = list(range(0, total, stride))[:num_bases]
    elif strategy == "highest_quality":
        scored = sorted(
            enumerate(train_data), key=lambda x: x[1]["gt_score"], reverse=True
        )
        indices = [idx for idx, _ in scored[:num_bases]]
    elif strategy == "stratified":
        import random

        rng = random.Random(seed)
        # Bin by MOS: [1-2), [2-3), [3-4), [4-5]
        bins: dict[int, list[int]] = {b: [] for b in range(4)}
        for idx, sample in enumerate(train_data):
            bin_idx = min(int(sample["gt_score"] - 1), 3)
            bins[bin_idx].append(idx)
        per_bin = num_bases // 4
        indices = []
        for b in range(4):
            candidates = bins[b]
            rng.shuffle(candidates)
            indices.extend(candidates[:per_bin])
        # Fill remainder from largest bin
        while len(indices) < num_bases:
            for b in range(4):
                remaining = [i for i in bins[b] if i not in indices]
                if remaining:
                    indices.append(remaining[0])
                    if len(indices) >= num_bases:
                        break
    else:
        msg = f"Unknown strategy: {strategy}"
        raise ValueError(msg)

    return [
        {
            "image_id": train_data[idx]["id"],
            "image_path": train_data[idx]["image"],
            "gt_score": train_data[idx]["gt_score"],
            "original_index": idx,
        }
        for idx in sorted(indices)
    ]


def _apply_degradation(
    image_arr: np.ndarray,
    profile: DegradationProfile,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply degradation effects to an image array.

    Returns the degraded image and IQA labels (severities + overall_quality).
    All severities in [0,1] where higher = more degraded.

    Args:
        image_arr: RGB uint8 array (H, W, 3).
        profile: Degradation profile with severity ranges.
        rng: Random instance for reproducible sampling.

    Returns:
        Tuple of (degraded_array, iqa_labels_dict).
    """
    img = image_arr.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    severities: dict[str, float] = {}

    # 1. Gaussian blur
    ksize = rng.randint(profile.blur_limit[0], profile.blur_limit[1])
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 1:
        from PIL import ImageFilter
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=ksize // 2))
        img = np.array(pil_img).astype(np.float32) / 255.0
    # Severity: normalize kernel size to [0,1] range (3→0.1, 21→1.0)
    severities["blur"] = min((ksize - 1) / 20.0, 1.0)

    # 2. Gaussian noise
    noise_var = rng.uniform(profile.noise_var[0], profile.noise_var[1])
    noise = np.random.RandomState(rng.randint(0, 2**31)).normal(
        0, np.sqrt(noise_var), img.shape
    ).astype(np.float32)
    img = np.clip(img + noise, 0, 1)
    severities["noise"] = min(noise_var / 0.10, 1.0)

    # 3. Brightness/contrast
    brightness = rng.uniform(profile.brightness_range[0], profile.brightness_range[1])
    contrast = rng.uniform(profile.contrast_range[0], profile.contrast_range[1])
    img = np.clip(contrast * img + brightness, 0, 1)
    # Paper degradation from brightness/contrast shifts
    severities["paper_degradation"] = min(
        abs(brightness) / 0.3 + abs(1.0 - contrast) / 0.7, 1.0
    )

    # 4. JPEG compression
    jpeg_q = rng.randint(profile.jpeg_quality[0], profile.jpeg_quality[1])
    import io
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_q)
    buf.seek(0)
    pil_img = Image.open(buf)
    img = np.array(pil_img).astype(np.float32) / 255.0
    severities["compression"] = max(0.0, (100 - jpeg_q) / 95.0)

    # 5. Rotation (geometric distortion)
    if profile.rotation_limit > 0:
        angle = rng.uniform(-profile.rotation_limit, profile.rotation_limit)
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
        img = np.array(pil_img).astype(np.float32) / 255.0
        severities["geometric_distortion"] = abs(angle) / 10.0
    else:
        severities["geometric_distortion"] = 0.0

    # No ink degradation or bleed-through in this pipeline
    severities["ink_degradation"] = 0.0
    severities["bleed_through"] = 0.0

    # Overall quality: 1 - max(severities)
    max_severity = max(severities.values())
    severities["overall_quality"] = max(0.0, 1.0 - max_severity)

    result = (img * 255).astype(np.uint8)
    return result, severities


def degrade_single_image(
    image: Image.Image,
    image_id: str,
    image_idx: int,
    base_seed: int = 20000,
    profiles: list[DegradationProfile] | None = None,
) -> list[dict]:
    """Apply degradation tiers to a single image.

    Args:
        image: PIL Image (RGB).
        image_id: Unique identifier.
        image_idx: Index for seed computation.
        base_seed: Base seed (different from OCR experiment to avoid collision).
        profiles: Degradation profiles. Default: all 4 DEGRADATION_PROFILES.

    Returns:
        List of dicts with: image_id, tier, degraded_image (PIL), iqa_labels (dict).
    """
    profiles = profiles or DEGRADATION_PROFILES
    image_arr = np.array(image)

    results = []
    for tier_idx, profile in enumerate(profiles):
        seed = base_seed + image_idx * 100 + tier_idx
        rng = random.Random(seed)

        degraded_arr, iqa_labels = _apply_degradation(image_arr, profile, rng)
        degraded_image = Image.fromarray(degraded_arr)

        results.append({
            "image_id": image_id,
            "tier": profile.tier_label,
            "degraded_image": degraded_image,
            "iqa_labels": iqa_labels,
            "seed": seed,
        })

    return results


def generate_stream1(
    train_json_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    num_bases: int = 350,
    base_seed: int = 20000,
    selection_strategy: str = "highest_quality",
    image_root: str | Path | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate Stream 1 controlled degradation samples.

    Full pipeline:
    1. Select base images from DIQA-5000 training set
    2. Apply 4 degradation levels per image
    3. Save degraded images and generate DeQA training records

    Args:
        train_json_path: Path to train_diqa_overall.json.
        data_root: Root directory for Data-DeQA-Score/ (for resolving image paths).
        output_dir: Directory for DIQA-5000_1 output (images + metadata).
        num_bases: Number of base images to degrade.
        base_seed: Base seed for deterministic degradation.
        selection_strategy: How to select base images.
        image_root: Override directory containing the actual image files.
            When set, image filenames are resolved from this directory
            instead of data_root / image_path. Useful when images are
            stored flat in a different location (e.g., research/vlm_calibration/data/res/).
        dry_run: If True, only select images and return plan without processing.

    Returns:
        List of DeQA training records (dicts).
    """
    train_json_path = Path(train_json_path)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    image_root_path = Path(image_root) if image_root else None

    # Step 1: Select base images
    bases = select_base_images(
        train_json_path, num_bases=num_bases, strategy=selection_strategy
    )
    logger.info("Selected %d base images for degradation", len(bases))

    if dry_run:
        logger.info("Dry run: would generate %d samples", len(bases) * len(DEGRADATION_PROFILES))
        return []

    # Step 2 & 3: Degrade and convert
    images_dir = output_dir / "images" / "stream1_degradation"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, base in enumerate(bases):
        # Resolve actual image path
        if image_root_path:
            filename = Path(base["image_path"]).name
            image_path = image_root_path / filename
        else:
            image_path = data_root / base["image_path"]

        if not image_path.exists():
            logger.warning("Image not found: %s (skipping)", image_path)
            continue

        image = Image.open(image_path).convert("RGB")
        degraded = degrade_single_image(
            image, base["image_id"], idx, base_seed=base_seed
        )

        for result in degraded:
            # Save degraded image
            tier = result["tier"]
            out_filename = f"{base['image_id'].replace('.jpg', '')}_{tier.lower()}.png"
            out_path = images_dir / out_filename
            result["degraded_image"].save(out_path, "PNG")

            # Convert to DeQA training record
            rel_path = f"DIQA-5000_1/images/stream1_degradation/{out_filename}"
            iqa = IQALabels(**result["iqa_labels"])

            record = iqa_to_deqa_record(
                image_id=f"s1_{base['image_id']}_{tier.lower()}",
                image_path=rel_path,
                iqa_labels=iqa,
                source="diqa_degradation",
                stream="stream1_degradation",
                sigma=0.8,
                weight=DETERMINISTIC_WEIGHT,
                seed=result["seed"],
            )
            record["degradation_tier"] = tier
            record["degradation_seed"] = result["seed"]
            record["base_image_id"] = base["image_id"]
            record["base_gt_score"] = base["gt_score"]
            records.append(record)

        if (idx + 1) % 50 == 0:
            logger.info(
                "Stream 1 progress: %d/%d bases processed (%d records)",
                idx + 1, len(bases), len(records),
            )

    logger.info(
        "Stream 1 complete: %d records from %d base images",
        len(records), len(bases),
    )
    return records
