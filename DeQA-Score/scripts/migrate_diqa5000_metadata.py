#!/usr/bin/env python
"""Migrate DIQA-5000 data into the unified master metadata JSONL.

Reads existing training JSONs, test manifest, image_detection metadata,
and VLM eval checkpoints, then produces a single master JSONL file with
one record per image.

Usage:
    python scripts/migrate_diqa5000_metadata.py \
        --output Data-DeQA-Score/DIQA/metadata/diqa5000_master.jsonl \
        [--image-detection-path /mnt/e/image_detection/metadata_registry/json/diqa-5000_metadata.json] \
        [--vlm-eval-dir results/vlm_teacher_eval/full_eval/checkpoints]
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path and patch src module to avoid CUDA imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if "src" not in sys.modules:
    src_module = types.ModuleType("src")
    src_module.__path__ = [str(project_root / "src")]
    sys.modules["src"] = src_module

from src.uncertainty.metadata_convert import (
    extract_canonical_id,
    from_diqa_test_json,
    from_diqa_training_json,
    from_image_detection_metadata,
    from_vlm_eval_jsonl,
)
from src.uncertainty.metadata_io import merge_records, write_master_jsonl
from src.uncertainty.metadata_schema import ImageMetadataRecord


def load_training_jsons(
    metas_dir: Path,
) -> dict[str, ImageMetadataRecord]:
    """Load per-dimension training JSONs and merge by canonical_id."""
    index: dict[str, ImageMetadataRecord] = {}

    dimension_files = {
        "overall": "train_diqa_overall.json",
        "sharpness": "train_diqa_sharpness.json",
        "color": "train_diqa_color.json",
    }

    for dimension, filename in dimension_files.items():
        filepath = metas_dir / filename
        if not filepath.exists():
            print(f"  [SKIP] {filepath} not found")
            continue

        with open(filepath) as f:
            records = json.load(f)

        metadata_list = from_diqa_training_json(records, dimension=dimension)
        print(f"  [OK] {filename}: {len(metadata_list)} records ({dimension})")

        for rec in metadata_list:
            if rec.canonical_id in index:
                index[rec.canonical_id] = merge_records(
                    index[rec.canonical_id], rec
                )
            else:
                index[rec.canonical_id] = rec

    return index


def load_test_manifest(
    metas_dir: Path,
    index: dict[str, ImageMetadataRecord],
) -> None:
    """Load test manifest and add entries not already in the index."""
    test_path = metas_dir / "diqa_test.json"
    if not test_path.exists():
        print(f"  [SKIP] {test_path} not found")
        return

    with open(test_path) as f:
        test_records = json.load(f)

    test_metadata = from_diqa_test_json(test_records)
    added = 0
    for rec in test_metadata:
        if rec.canonical_id not in index:
            index[rec.canonical_id] = rec
            added += 1
        else:
            # Merge in test paths if training record exists
            index[rec.canonical_id] = merge_records(
                index[rec.canonical_id], rec
            )
    print(f"  [OK] diqa_test.json: {len(test_metadata)} records, {added} new test-only images")


def load_image_detection(
    metadata_path: Path,
    index: dict[str, ImageMetadataRecord],
) -> int:
    """Load image_detection metadata and attach DocumentContext snapshots."""
    if not metadata_path.exists():
        print(f"  [SKIP] {metadata_path} not found")
        return 0

    print(f"  Loading {metadata_path} (this may take a moment)...")
    with open(metadata_path) as f:
        data = json.load(f)

    samples = data.get("samples", [])
    matched = 0

    for sample in samples:
        source = sample.get("source", {})
        original_filename = source.get("original_filename", "")
        if not original_filename:
            continue

        try:
            canonical_id = extract_canonical_id(original_filename)
        except ValueError:
            continue

        if canonical_id not in index:
            continue

        doc_ctx, sample_id = from_image_detection_metadata(sample)
        if doc_ctx is not None:
            existing = index[canonical_id]
            # Build a partial record with document context
            update = ImageMetadataRecord(
                canonical_id=canonical_id,
                dataset=existing.dataset,
                image_path_res=existing.image_path_res,
                image_detection_id=sample_id,
                document=doc_ctx,
            )
            index[canonical_id] = merge_records(existing, update)
            matched += 1

    print(f"  [OK] image_detection: {matched}/{len(index)} images matched")
    return matched


def load_vlm_evals(
    vlm_dir: Path,
    index: dict[str, ImageMetadataRecord],
) -> int:
    """Load VLM eval checkpoints and attach to matching records."""
    if not vlm_dir.exists():
        print(f"  [SKIP] {vlm_dir} not found")
        return 0

    jsonl_files = sorted(vlm_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"  [SKIP] No JSONL files in {vlm_dir}")
        return 0

    total_attached = 0
    for jsonl_path in jsonl_files:
        vlm_data = from_vlm_eval_jsonl(jsonl_path)
        attached = 0

        for canonical_id, eval_records in vlm_data.items():
            if canonical_id not in index:
                continue

            existing = index[canonical_id]
            update = ImageMetadataRecord(
                canonical_id=canonical_id,
                dataset=existing.dataset,
                image_path_res=existing.image_path_res,
                vlm_evals=eval_records,
            )
            index[canonical_id] = merge_records(existing, update)
            attached += 1

        model_name = jsonl_path.stem
        print(f"  [OK] {model_name}: {attached} images")
        total_attached += attached

    return total_attached


def print_summary(index: dict[str, ImageMetadataRecord]) -> None:
    """Print migration summary statistics."""
    records = list(index.values())
    total = len(records)

    split_counts: dict[str, int] = defaultdict(int)
    dim_counts: dict[str, int] = defaultdict(int)
    doc_count = 0
    vlm_count = 0

    for rec in records:
        split_counts[rec.split or "unknown"] += 1
        for dim in rec.dimensions:
            dim_counts[dim] += 1
        if rec.document is not None:
            doc_count += 1
        if rec.vlm_evals:
            vlm_count += 1

    print(f"\n{'='*50}")
    print(f"Migration Summary")
    print(f"{'='*50}")
    print(f"Total records:       {total}")
    print(f"Splits:              {dict(split_counts)}")
    print(f"Dimension coverage:  {dict(dim_counts)}")
    print(f"Document context:    {doc_count}/{total}")
    print(f"VLM evaluations:     {vlm_count}/{total}")
    print(f"Schema version:      {records[0].schema_version if records else 'N/A'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate DIQA-5000 data to unified metadata JSONL"
    )
    parser.add_argument(
        "--metas-dir",
        type=Path,
        default=project_root / "Data-DeQA-Score" / "DIQA" / "metas",
        help="Directory containing DIQA training JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Data-DeQA-Score" / "DIQA" / "metadata" / "diqa5000_master.jsonl",
        help="Output master JSONL path",
    )
    parser.add_argument(
        "--image-detection-path",
        type=Path,
        default=Path("/mnt/e/image_detection/metadata_registry/json/diqa-5000_metadata.json"),
        help="Path to image_detection master metadata JSON",
    )
    parser.add_argument(
        "--vlm-eval-dir",
        type=Path,
        default=project_root.parent / "results" / "vlm_teacher_eval" / "full_eval" / "checkpoints",
        help="Directory containing VLM eval JSONL checkpoints",
    )
    parser.add_argument(
        "--export-schema",
        action="store_true",
        default=True,
        help="Export JSON Schema alongside the master JSONL",
    )
    args = parser.parse_args()

    print("Step 1: Loading training JSONs...")
    index = load_training_jsons(args.metas_dir)

    print("\nStep 2: Loading test manifest...")
    load_test_manifest(args.metas_dir, index)

    print("\nStep 3: Loading image_detection metadata...")
    load_image_detection(args.image_detection_path, index)

    print("\nStep 4: Loading VLM eval checkpoints...")
    load_vlm_evals(args.vlm_eval_dir, index)

    # Sort by canonical_id for deterministic output
    sorted_records = sorted(index.values(), key=lambda r: r.canonical_id)

    print(f"\nStep 5: Writing {len(sorted_records)} records to {args.output}...")
    count = write_master_jsonl(sorted_records, args.output)
    print(f"  [OK] Wrote {count} records")

    # Export JSON Schema
    if args.export_schema:
        schema_path = args.output.parent / "schema.json"
        schema = ImageMetadataRecord.schema_json(indent=2)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(schema)
        print(f"  [OK] Exported JSON Schema to {schema_path}")

    print_summary(index)


if __name__ == "__main__":
    main()
