"""Stream 2: Fresh synthetic document generation with degradation.

Generates pristine multi-script documents using the image_detection
renderer (with fixed font diversity), then applies degradation tiers
to produce document quality variants for DIQA training.

This replaces the original GCS-based approach because synth-multiscript-v3
had a font diversity bug (26/27 scripts used only 1 font).

Target: 2,000 samples across 10+ scripts with quality tier distribution:
    PRISTINE 10%, HIGH 25%, MEDIUM 35%, LOW 20%, DEGRADED 10%.

Requires: image_detection venv with image_preprocessing_detector installed.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .iqa_to_deqa import iqa_to_deqa_record
from .stream1_degradation import (
    DEGRADATION_PROFILES,
    DegradationProfile,
    _apply_degradation,
)

logger = logging.getLogger(__name__)

# Training weight for deterministic labels
DETERMINISTIC_WEIGHT = 0.7

# Target quality tier distribution
TIER_DISTRIBUTION = {
    "pristine": 0.10,
    "light": 0.25,
    "moderate": 0.35,
    "heavy": 0.20,
    "historical": 0.10,
}

# Pristine profile (minimal degradation)
_PRISTINE_PROFILE = DegradationProfile(
    name="pristine",
    tier_label="PRISTINE",
    blur_limit=(1, 3),
    noise_var=(0.0, 0.001),
    jpeg_quality=(90, 98),
    brightness_range=(-0.02, 0.02),
    contrast_range=(0.95, 1.05),
    rotation_limit=0,
    quality_range=(0.95, 1.0),
)

# Map tier names to profiles
_PROFILE_MAP: dict[str, DegradationProfile] = {
    "pristine": _PRISTINE_PROFILE,
    **{p.name: p for p in DEGRADATION_PROFILES},
}

# Scripts to generate (exclude OOD-only: Armn, Geor; exclude Hang: no corpus)
GENERATION_SCRIPTS = [
    "Arab", "Beng", "Cyrl", "Deva", "Ethi", "Grek", "Gujr", "Guru",
    "Hans", "Hant", "Hebr", "Jpan", "Khmr", "Knda",
    "Laoo", "Latn", "Mlym", "Mymr", "Orya", "Sinh", "Taml",
    "Telu", "Thai", "Tibt",
]

# Path to image_detection venv python
IMAGE_DETECTION_PYTHON = Path(
    "/home/byron/dev/image_detection/.venv/bin/python"
)

# Helper script that runs inside image_detection venv
_GENERATE_HELPER = "_stream2_generate_pristine.py"


def plan_tier_assignments(
    total_samples: int,
    tier_distribution: dict[str, float] | None = None,
    seed: int = 42,
) -> list[str]:
    """Plan which degradation tier to apply to each sample.

    Produces a shuffled list of tier assignments matching the target
    distribution as closely as possible.

    Args:
        total_samples: Number of samples to assign.
        tier_distribution: Target distribution. Default: TIER_DISTRIBUTION.
        seed: Random seed.

    Returns:
        List of tier names, length = total_samples.
    """
    dist = tier_distribution or TIER_DISTRIBUTION
    assignments = []
    for tier, pct in dist.items():
        count = round(total_samples * pct)
        assignments.extend([tier] * count)

    # Adjust for rounding
    while len(assignments) < total_samples:
        assignments.append("moderate")
    assignments = assignments[:total_samples]

    rng = random.Random(seed)
    rng.shuffle(assignments)
    return assignments


def _write_generation_helper(helper_path: Path, config: dict) -> None:
    """Write a standalone Python script that generates pristine images.

    This script runs inside the image_detection venv where the
    MultiScriptDocumentGenerator is available.

    Args:
        helper_path: Path to write the helper script.
        config: Generation config dict (scripts, samples_per_script, etc).
    """
    config_json = json.dumps(config)
    script = f'''#!/usr/bin/env python3
"""Auto-generated helper: generates pristine documents via image_detection."""
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

config = json.loads({config_json!r})

from image_preprocessing_detector.synthetic.generator import (
    GenerationConfig, MultiScriptDocumentGenerator,
)

output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

gen_config = GenerationConfig(
    scripts=config["scripts"],
    samples_per_script=config["samples_per_script"],
    output_dir=None,
    save_images=False,
    save_metadata=False,
    image_format="png",
    seed=config["seed"],
    pristine_ratio=1.0,  # All pristine — we apply degradation ourselves
    dpi=150,  # Half resolution for speed (still 1240x1754 — fine for DIQA)
    augmenter="albumentations",
    color_mode_enabled=False,
    skew_augmentation=False,
    orientation_augmentation=False,
)

generator = MultiScriptDocumentGenerator(config=gen_config)
logger.info("Initializing generator...")
ok = generator.initialize(download_corpus=False, scan_fonts=True)
if not ok:
    logger.error("Generator initialization failed")
    sys.exit(1)

manifest = []
count = 0
start = time.time()

for sample in generator.generate():
    primary_script = sorted(sample.scripts)[0]
    filename = f"pristine_{{primary_script}}_{{count:05d}}.png"
    out_path = output_dir / filename

    sample.image.save(out_path, "PNG")

    meta = {{
        "filename": filename,
        "sample_id": sample.sample_id,
        "scripts": list(sample.scripts),
        "primary_script": primary_script,
        "layout_type": sample.layout_type.value if hasattr(sample.layout_type, "value") else str(sample.layout_type),
        "resolution_tier": getattr(sample, "resolution_tier", "STANDARD"),
        "font_families_used": getattr(sample, "font_families_used", []),
    }}
    manifest.append(meta)
    count += 1

    if count % 100 == 0:
        elapsed = time.time() - start
        logger.info("Generated %d/%d pristine images (%.1f img/s)",
                     count, config["total"], count / elapsed if elapsed > 0 else 0)

manifest_path = output_dir / "pristine_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

elapsed = time.time() - start
logger.info("Done: %d pristine images in %.1fs", count, elapsed)
print(json.dumps({{"count": count, "manifest": str(manifest_path)}}))
'''
    helper_path.write_text(script)
    helper_path.chmod(0o755)


def generate_pristine_images(
    output_dir: Path,
    scripts: list[str],
    total_samples: int,
    seed: int = 30000,
) -> list[dict]:
    """Generate pristine document images using image_detection renderer.

    Spawns a subprocess using the image_detection venv to access the
    MultiScriptDocumentGenerator with fixed font diversity.

    Args:
        output_dir: Directory to save pristine PNG images.
        scripts: ISO 15924 script codes to include.
        total_samples: Total number of pristine images to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of manifest entries (dicts with filename, script, etc).

    Raises:
        RuntimeError: If generation subprocess fails.
    """
    if not IMAGE_DETECTION_PYTHON.exists():
        msg = f"image_detection venv not found at {IMAGE_DETECTION_PYTHON}"
        raise RuntimeError(msg)

    pristine_dir = output_dir / "pristine_staging"
    pristine_dir.mkdir(parents=True, exist_ok=True)

    # Calculate samples per script (balanced)
    samples_per_script = total_samples // len(scripts)

    config = {
        "scripts": scripts,
        "samples_per_script": samples_per_script,
        "total": total_samples,
        "seed": seed,
        "output_dir": str(pristine_dir),
    }

    # Write helper to /tmp to avoid NTFS chmod issues on E: drive
    helper_path = Path("/tmp") / _GENERATE_HELPER
    _write_generation_helper(helper_path, config)

    logger.info(
        "Generating %d pristine images across %d scripts (subprocess)...",
        total_samples, len(scripts),
    )

    result = subprocess.run(
        [str(IMAGE_DETECTION_PYTHON), str(helper_path)],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
        cwd=str(Path("/home/byron/dev/image_detection")),
    )

    if result.returncode != 0:
        logger.error("Pristine generation failed:\n%s", result.stderr[-2000:])
        msg = f"Pristine generation subprocess failed (exit {result.returncode})"
        raise RuntimeError(msg)

    # Parse output
    stdout_lines = result.stdout.strip().split("\n")
    status = json.loads(stdout_lines[-1])
    logger.info("Pristine generation complete: %d images", status["count"])

    # Load manifest
    manifest_path = pristine_dir / "pristine_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Clean up helper script
    helper_path.unlink(missing_ok=True)

    return manifest


def generate_stream2(
    output_dir: str | Path,
    total_samples: int = 2000,
    base_seed: int = 30000,
    scripts: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate Stream 2 fresh synthetic document samples.

    Full pipeline:
    1. Generate pristine documents via image_detection renderer
    2. Assign degradation tiers per target distribution
    3. Apply degradation and generate DeQA training records

    Args:
        output_dir: Directory for DIQA-5000_1 output.
        total_samples: Total samples to generate.
        base_seed: Base seed for deterministic processing.
        scripts: Script codes to use (default: GENERATION_SCRIPTS).
        dry_run: If True, plan only without processing images.

    Returns:
        List of DeQA training records (dicts).
    """
    output_dir = Path(output_dir)
    scripts = scripts or GENERATION_SCRIPTS

    # Step 1: Plan tier assignments
    tier_assignments = plan_tier_assignments(total_samples, seed=base_seed)

    if dry_run:
        from collections import Counter

        tier_counts = Counter(tier_assignments)
        per_script = total_samples // len(scripts)
        logger.info("Dry run plan:")
        logger.info("  Total samples: %d", total_samples)
        logger.info("  Scripts: %d (%d per script)", len(scripts), per_script)
        logger.info("  Tier distribution: %s", dict(tier_counts))
        return []

    # Step 2: Generate pristine images
    start_time = time.time()
    manifest = generate_pristine_images(
        output_dir=output_dir,
        scripts=scripts,
        total_samples=total_samples,
        seed=base_seed,
    )

    if len(manifest) < total_samples:
        logger.warning(
            "Generated %d pristine images (target: %d), adjusting...",
            len(manifest), total_samples,
        )
        tier_assignments = tier_assignments[: len(manifest)]

    pristine_dir = output_dir / "pristine_staging"
    images_dir = output_dir / "images" / "stream2_synth_multiscript"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Apply degradation and create records
    records = []
    for idx, (entry, tier_name) in enumerate(zip(manifest, tier_assignments)):
        seed = base_seed + idx
        profile = _PROFILE_MAP.get(tier_name)
        if profile is None:
            logger.warning("Unknown tier %s, skipping", tier_name)
            continue

        pristine_path = pristine_dir / entry["filename"]
        try:
            image = Image.open(pristine_path).convert("RGB")
        except Exception:
            logger.warning("Failed to open %s, skipping", pristine_path)
            continue

        rng = random.Random(seed)
        image_arr = np.array(image)
        degraded_arr, iqa_labels = _apply_degradation(image_arr, profile, rng)
        degraded_image = Image.fromarray(degraded_arr)

        # Save degraded image
        script = entry["primary_script"]
        out_filename = f"synth_{script}_{idx:05d}_{tier_name}.png"
        out_path = images_dir / out_filename
        degraded_image.save(out_path, "PNG")

        # Convert to DeQA record
        rel_path = f"DIQA-5000_1/images/stream2_synth_multiscript/{out_filename}"

        record = iqa_to_deqa_record(
            image_id=f"s2_synth_{script}_{idx:05d}",
            image_path=rel_path,
            iqa_labels=iqa_labels,
            source="synth_fresh_generation",
            stream="stream2_synth_multiscript",
            sigma=0.8,
            weight=DETERMINISTIC_WEIGHT,
            seed=seed,
        )
        record["script"] = script
        record["degradation_tier"] = tier_name
        record["degradation_seed"] = seed
        record["layout_type"] = entry.get("layout_type", "")
        record["font_families_used"] = entry.get("font_families_used", [])
        records.append(record)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Stream 2 progress: %d/%d processed (%.1fs)",
                idx + 1, len(manifest), elapsed,
            )

    logger.info(
        "Stream 2 complete: %d records from %d scripts",
        len(records),
        len({r["script"] for r in records}),
    )
    return records
