"""Tier 1 orchestrator: build DIQA-5000_1 from three expansion streams.

Combines Stream 1 (controlled degradation), Stream 2 (synth-multiscript-v3),
and Stream 3 (VLM consensus) with the DIQA-5000_0 base to produce the
complete DIQA-5000_1 training dataset.

Output structure:
    Data-DeQA-Score/DIQA-5000_1/
        manifest.json                    # Dataset provenance & stats
        train_overall.json               # Combined training data (8,990 samples)
        holdout_overall.json             # 10% holdout from new data (~550)
        vlm_consensus_results.json       # Raw VLM labels for audit
        images/
            stream1_degradation/         # 1,400 degraded DIQA images
            stream2_synth_multiscript/   # 2,000 synth-multiscript images
            stream3_ohr_bench/           # VLM-labeled real documents
            stream3_tobacco800/
            stream3_smartdoc_qa/
            stream3_realdae/
            stream3_funsd_plus/
            stream3_ocr_quality/
            stream3_sroie/

Usage:
    python -m src.expansion.build_tier1 --config tier1_config.json
    python -m src.expansion.build_tier1 --dry-run    # Plan without processing
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .iqa_to_deqa import LEVEL_NAMES
from .manifest import (
    DatasetManifest,
    SourceEntry,
    compute_quality_distribution,
)

logger = logging.getLogger(__name__)


@dataclass
class Tier1Config:
    """Configuration for Tier 1 dataset expansion."""

    # Paths
    data_root: str = "Data-DeQA-Score"
    base_train_json: str = "Data-DeQA-Score/DIQA/metas/train_diqa_overall.json"
    output_dir: str = "Data-DeQA-Score/DIQA-5000_1"

    # Stream 1: Controlled degradation
    stream1_enabled: bool = True
    stream1_num_bases: int = 350
    stream1_selection: str = "highest_quality"
    stream1_seed: int = 20000
    stream1_image_root: str = ""  # Override for DIQA image directory

    # Stream 2: Synth-multiscript-v3
    stream2_enabled: bool = True
    stream2_synth_root: str = ""  # Must be set (local path or post-download)
    stream2_total: int = 2000
    stream2_seed: int = 30000

    # Stream 3: VLM consensus
    stream3_enabled: bool = True
    stream3_api_key: str = ""  # Set via env or config
    stream3_seed: int = 40000
    stream3_sources: list[dict] = field(default_factory=list)
    # Each source: {name, image_dir, count, has_text_gt, glob_pattern}

    # Training
    holdout_pct: float = 0.10  # 10% holdout from expansion data
    base_weight: float = 1.0
    deterministic_weight: float = 0.7
    vlm_weight: float = 0.5
    global_seed: int = 42

    # Validation gates (Tier 1)
    gate_id_wsrcc_min: float = -0.01  # relative to baseline
    gate_ood_wsrcc_min: float = 0.02  # relative to baseline
    gate_tail_srcc_min: float = 0.50
    gate_level_mass_pct: float = 2.0  # all 5 levels must have > 2% mass

    @classmethod
    def from_json(cls, path: str | Path) -> Tier1Config:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


def create_diqa5000_0_manifest(
    base_train_json: str | Path,
    output_dir: str | Path,
) -> DatasetManifest:
    """Create the DIQA-5000_0 baseline manifest from existing training data.

    Args:
        base_train_json: Path to train_diqa_overall.json.
        output_dir: Directory for DIQA-5000_0 output.

    Returns:
        DatasetManifest for the baseline.
    """
    with open(base_train_json) as f:
        base_data = json.load(f)

    mos_scores = [s["gt_score"] for s in base_data]

    manifest = DatasetManifest(
        version="DIQA-5000_0",
        tier=0,
        description=(
            "DIQA-5000 baseline training set. 350 base document images × 10 "
            "quality variants = 3,500 samples with human ground truth labels."
        ),
        total_samples=len(base_data),
        quality_distribution=compute_quality_distribution(mos_scores),
        training_files={"overall": "train_diqa_overall.json"},
        validation_gates={},
    )
    manifest.add_source(SourceEntry(
        name="diqa_5000_human_gt",
        stream="base",
        count=len(base_data),
        label_method="human_gt",
        weight=1.0,
        description="DIQA-5000 training split with human MOS annotations",
    ))
    # Don't double-count: reset since add_source incremented
    manifest.total_samples = len(base_data)
    manifest.new_samples = len(base_data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.save(output_dir)
    logger.info("Created DIQA-5000_0 manifest: %d samples", len(base_data))
    return manifest


def split_holdout(
    records: list[dict],
    holdout_pct: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split expansion records into train and holdout sets.

    Stratified by source to ensure holdout has representative coverage.

    Args:
        records: List of training records from expansion streams.
        holdout_pct: Fraction to reserve for holdout.
        seed: Random seed.

    Returns:
        Tuple of (train_records, holdout_records).
    """
    rng = random.Random(seed)

    # Group by source
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        source = rec.get("source", "unknown")
        by_source.setdefault(source, []).append(rec)

    train_records = []
    holdout_records = []

    for source, source_records in by_source.items():
        rng.shuffle(source_records)
        n_holdout = max(1, int(len(source_records) * holdout_pct))
        holdout_records.extend(source_records[:n_holdout])
        train_records.extend(source_records[n_holdout:])

    logger.info(
        "Split: %d train, %d holdout (%.1f%%)",
        len(train_records), len(holdout_records),
        len(holdout_records) / max(len(records), 1) * 100,
    )
    return train_records, holdout_records


def validate_level_distribution(records: list[dict], min_mass_pct: float = 2.0) -> bool:
    """Check that all 5 quality levels have sufficient representation.

    Args:
        records: Training records with level_probs.
        min_mass_pct: Minimum percentage of total mass per level.

    Returns:
        True if all levels pass the gate.
    """
    if not records:
        return False

    total_mass = np.zeros(5)
    for rec in records:
        total_mass += np.array(rec["level_probs"])

    total_mass /= total_mass.sum()
    pcts = total_mass * 100

    passed = True
    for i, name in enumerate(LEVEL_NAMES):
        if pcts[i] < min_mass_pct:
            logger.warning(
                "Level '%s' has %.1f%% mass (minimum: %.1f%%)",
                name, pcts[i], min_mass_pct,
            )
            passed = False
        else:
            logger.debug("Level '%s': %.1f%%", name, pcts[i])

    return passed


def merge_training_data(
    base_train_json: str | Path,
    expansion_records: list[dict],
    output_path: str | Path,
) -> int:
    """Merge base DIQA training data with expansion records.

    Args:
        base_train_json: Path to existing train_diqa_overall.json.
        expansion_records: New records from expansion streams.
        output_path: Output path for merged training JSON.

    Returns:
        Total number of samples in merged dataset.
    """
    with open(base_train_json) as f:
        base_data = json.load(f)

    # Tag base data for tracking
    for rec in base_data:
        if "source" not in rec:
            rec["source"] = "diqa_5000_human_gt"
            rec["stream"] = "base"
            rec["confidence_weight"] = 1.0
            rec["pseudo_label"] = False

    merged = base_data + expansion_records

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(
        "Merged training data: %d base + %d expansion = %d total",
        len(base_data), len(expansion_records), len(merged),
    )
    return len(merged)


def build_tier1(config: Tier1Config) -> DatasetManifest:
    """Execute the full Tier 1 dataset expansion pipeline.

    Orchestrates all three streams, applies holdout split, merges with
    base data, and generates the DIQA-5000_1 manifest.

    Args:
        config: Tier 1 configuration.

    Returns:
        DatasetManifest for DIQA-5000_1.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config.save(output_dir / "tier1_config.json")

    all_expansion_records = []

    # ── Stream 1: Controlled Degradation ──────────────────────────────────
    if config.stream1_enabled:
        logger.info("=== Stream 1: Controlled Degradation ===")
        from .stream1_degradation import generate_stream1

        s1_records = generate_stream1(
            train_json_path=config.base_train_json,
            data_root=config.data_root,
            output_dir=output_dir,
            num_bases=config.stream1_num_bases,
            base_seed=config.stream1_seed,
            selection_strategy=config.stream1_selection,
            image_root=config.stream1_image_root or None,
        )
        all_expansion_records.extend(s1_records)
        logger.info("Stream 1: %d records", len(s1_records))

    # ── Stream 2: Synth-Multiscript-v3 ────────────────────────────────────
    if config.stream2_enabled and config.stream2_synth_root:
        logger.info("=== Stream 2: Synth-Multiscript-v3 ===")
        from .stream2_synth_multiscript import generate_stream2

        s2_records = generate_stream2(
            synth_root=config.stream2_synth_root,
            output_dir=output_dir,
            total_samples=config.stream2_total,
            base_seed=config.stream2_seed,
        )
        all_expansion_records.extend(s2_records)
        logger.info("Stream 2: %d records", len(s2_records))

    # ── Stream 3: VLM Consensus ───────────────────────────────────────────
    if config.stream3_enabled:
        logger.info("=== Stream 3: VLM Consensus ===")
        from .stream3_vlm_consensus import TIER1_SOURCES, SourceDataset, generate_stream3

        # Use configured sources if provided, else default TIER1_SOURCES
        if config.stream3_sources:
            sources = [
                SourceDataset(
                    name=s["name"],
                    image_dir=Path(s["image_dir"]),
                    count=s.get("count", 500),
                    has_text_gt=s.get("has_text_gt", False),
                    glob_patterns=tuple(s.get("glob_patterns", ("*.png", "*.jpg", "*.jpeg"))),
                    split_filter=s.get("split_filter"),
                    exclude_suffix=s.get("exclude_suffix"),
                    recursive=s.get("recursive", False),
                )
                for s in config.stream3_sources
            ]
        else:
            sources = list(TIER1_SOURCES)

        s3_records = generate_stream3(
            sources=sources,
            output_dir=output_dir,
            api_key=config.stream3_api_key,
            base_seed=config.stream3_seed,
        )
        all_expansion_records.extend(s3_records)
        logger.info("Stream 3: %d records", len(s3_records))

    logger.info("Total expansion records: %d", len(all_expansion_records))

    # ── Holdout Split ─────────────────────────────────────────────────────
    train_records, holdout_records = split_holdout(
        all_expansion_records,
        holdout_pct=config.holdout_pct,
        seed=config.global_seed,
    )

    # Save holdout
    holdout_path = output_dir / "holdout_overall.json"
    with open(holdout_path, "w") as f:
        json.dump(holdout_records, f, indent=2)
    logger.info("Saved %d holdout records", len(holdout_records))

    # ── Merge with Base ───────────────────────────────────────────────────
    train_path = output_dir / "train_overall.json"
    total = merge_training_data(
        config.base_train_json, train_records, train_path
    )

    # ── Validate Level Distribution ───────────────────────────────────────
    with open(train_path) as f:
        all_training = json.load(f)
    level_gate = validate_level_distribution(
        all_training, min_mass_pct=config.gate_level_mass_pct
    )
    if not level_gate:
        logger.warning("VALIDATION GATE FAILED: level distribution imbalance")

    # ── Build Manifest ────────────────────────────────────────────────────
    mos_scores = [r["gt_score"] for r in all_training]

    manifest = DatasetManifest(
        version="DIQA-5000_1",
        tier=1,
        parent_version="DIQA-5000_0",
        description=(
            "Tier 1 expansion: quality balance + domain seed. "
            f"{len(train_records)} new samples from 3 streams + "
            f"3,500 DIQA-5000 base = {total} total."
        ),
        total_samples=total,
        new_samples=len(train_records),
        holdout_pct=config.holdout_pct,
        quality_distribution=compute_quality_distribution(mos_scores),
        training_files={"overall": "train_overall.json"},
        base_upsampling=1,
        vlm_models=["google/gemini-3.1-flash-lite-preview", "qwen/qwen3.5-122b-a10b"],
        validation_gates={
            "id_wsrcc_delta_min": config.gate_id_wsrcc_min,
            "ood_wsrcc_delta_min": config.gate_ood_wsrcc_min,
            "tail_srcc_min": config.gate_tail_srcc_min,
            "level_mass_pct_min": config.gate_level_mass_pct,
            "level_gate_passed": level_gate,
        },
    )

    # Add source entries
    stream_counts: dict[str, int] = {}
    for rec in train_records:
        stream = rec.get("stream", "unknown")
        stream_counts[stream] = stream_counts.get(stream, 0) + 1

    if "stream1_degradation" in stream_counts:
        manifest.add_source(SourceEntry(
            name="diqa_controlled_degradation",
            stream="stream1_degradation",
            count=stream_counts["stream1_degradation"],
            label_method="deterministic",
            weight=config.deterministic_weight,
            description="4 degradation levels applied to DIQA-5000 base images",
        ))

    if "stream2_synth_multiscript" in stream_counts:
        manifest.add_source(SourceEntry(
            name="synth_multiscript_v3",
            stream="stream2_synth_multiscript",
            count=stream_counts["stream2_synth_multiscript"],
            label_method="deterministic",
            weight=config.deterministic_weight,
            description="Degradation replay on synth-multiscript-v3 images",
        ))

    vlm_sources = [k for k in stream_counts if k.startswith("stream3")]
    for src_stream in vlm_sources:
        src_name = src_stream.replace("stream3_", "")
        manifest.add_source(SourceEntry(
            name=src_name,
            stream=src_stream,
            count=stream_counts[src_stream],
            label_method="vlm_consensus",
            weight=config.vlm_weight,
            description=f"VLM consensus labels from {src_name}",
        ))

    # Base data (not double-counted in new_samples)
    manifest.sources.insert(0, SourceEntry(
        name="diqa_5000_human_gt",
        stream="base",
        count=3500,
        label_method="human_gt",
        weight=config.base_weight,
        description="DIQA-5000 original training data",
    ))

    manifest.save(output_dir)
    logger.info("Manifest saved to %s", output_dir / "manifest.json")
    logger.info("\n%s", manifest.summary())

    return manifest


def main() -> None:
    """CLI entry point for Tier 1 dataset generation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build DIQA-5000_1 (Tier 1 expansion)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to tier1_config.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan only, no image processing or API calls",
    )
    parser.add_argument(
        "--stream", type=str, default=None,
        choices=["stream1", "stream2", "stream3"],
        help="Run only a specific stream",
    )
    args = parser.parse_args()

    if args.config:
        config = Tier1Config.from_json(args.config)
    else:
        config = Tier1Config()

    if args.stream:
        config.stream1_enabled = args.stream == "stream1"
        config.stream2_enabled = args.stream == "stream2"
        config.stream3_enabled = args.stream == "stream3"

    if args.dry_run:
        logger.info("DRY RUN MODE — no images will be processed")
        # Run stream generators in dry-run mode individually
        if config.stream1_enabled:
            from .stream1_degradation import generate_stream1
            generate_stream1(
                config.base_train_json, config.data_root, config.output_dir,
                num_bases=config.stream1_num_bases, dry_run=True,
            )
        if config.stream2_enabled and config.stream2_synth_root:
            from .stream2_synth_multiscript import generate_stream2
            generate_stream2(
                config.stream2_synth_root, config.output_dir,
                total_samples=config.stream2_total, dry_run=True,
            )
        if config.stream3_enabled:
            from .stream3_vlm_consensus import generate_stream3
            generate_stream3(output_dir=config.output_dir, dry_run=True)
        return

    manifest = build_tier1(config)
    print(manifest.summary())


if __name__ == "__main__":
    main()
