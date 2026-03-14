"""JSONL I/O and merge logic for master metadata files.

Provides read/write/streaming access to master JSONL files and a merge
function for combining partial records (e.g., different dimensions or
pipeline layers) into unified per-image records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from .metadata_schema import (
    DimensionRecord,
    ImageMetadataRecord,
    VLMEvalRecord,
)


def write_master_jsonl(
    records: List[ImageMetadataRecord],
    path: Union[str, Path],
    mode: str = "w",
) -> int:
    """Write metadata records to a JSONL file.

    Args:
        records: List of ImageMetadataRecord instances.
        path: Output file path.
        mode: File open mode ('w' for overwrite, 'a' for append).

    Returns:
        Number of records written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode) as f:
        for record in records:
            f.write(record.json() + "\n")

    return len(records)


def read_master_jsonl(path: Union[str, Path]) -> List[ImageMetadataRecord]:
    """Read all metadata records from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of ImageMetadataRecord instances.
    """
    records = []
    for record in read_master_jsonl_lazy(path):
        records.append(record)
    return records


def read_master_jsonl_lazy(
    path: Union[str, Path],
) -> Iterator[ImageMetadataRecord]:
    """Lazily iterate over metadata records from a JSONL file.

    Args:
        path: Path to JSONL file.

    Yields:
        ImageMetadataRecord instances, one per line.
    """
    path = Path(path)
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ImageMetadataRecord.parse_raw(line)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse line {line_num} in {path}: {e}"
                ) from e


def build_index(
    path: Union[str, Path],
) -> Dict[str, ImageMetadataRecord]:
    """Build an index of records keyed by canonical_id.

    Args:
        path: Path to JSONL file.

    Returns:
        Dict mapping canonical_id to ImageMetadataRecord.
    """
    index: Dict[str, ImageMetadataRecord] = {}
    for record in read_master_jsonl_lazy(path):
        index[record.canonical_id] = record
    return index


def merge_records(
    existing: ImageMetadataRecord,
    update: ImageMetadataRecord,
) -> ImageMetadataRecord:
    """Merge two partial records for the same image.

    The ``update`` record's non-None fields take precedence over
    ``existing`` for top-level optional fields. Dimensions are merged
    by key (update wins for same dimension). VLM evals are merged by
    model_id (update wins for duplicates, new ones appended). Tags are
    merged with update taking precedence.

    Args:
        existing: The base record.
        update: The record with new data to merge in.

    Returns:
        A new merged ImageMetadataRecord.

    Raises:
        ValueError: If canonical_ids don't match.
    """
    if existing.canonical_id != update.canonical_id:
        raise ValueError(
            f"Cannot merge records with different canonical_ids: "
            f"{existing.canonical_id} vs {update.canonical_id}"
        )

    # Start from existing as base
    data = existing.dict()

    # Merge top-level optional fields (update wins if not None)
    for field_name in (
        "split",
        "image_path_ori",
        "image_detection_id",
        "document",
        "ood",
        "spread",
        "pipeline_run_id",
    ):
        update_val = getattr(update, field_name)
        if update_val is not None:
            if isinstance(update_val, (dict, list)):
                data[field_name] = update_val
            elif hasattr(update_val, "dict"):
                data[field_name] = update_val.dict()
            else:
                data[field_name] = update_val

    # Merge VLM evals by model_id
    existing_vlm: Dict[str, dict] = {}
    for ev in existing.vlm_evals:
        existing_vlm[ev.model_id] = ev.dict()
    for ev in update.vlm_evals:
        existing_vlm[ev.model_id] = ev.dict()
    data["vlm_evals"] = list(existing_vlm.values())

    # Merge dimensions by key
    merged_dims: Dict[str, dict] = {}
    for dim_key, dim_rec in existing.dimensions.items():
        merged_dims[dim_key] = dim_rec.dict()
    for dim_key, dim_rec in update.dimensions.items():
        if dim_key in merged_dims:
            merged_dims[dim_key] = _merge_dimension(
                merged_dims[dim_key], dim_rec.dict()
            )
        else:
            merged_dims[dim_key] = dim_rec.dict()
    data["dimensions"] = merged_dims

    # Merge boolean flags (True wins)
    if update.is_pseudo_labeled:
        data["is_pseudo_labeled"] = True

    # Merge tags (update wins)
    merged_tags = dict(existing.tags)
    merged_tags.update(update.tags)
    data["tags"] = merged_tags

    # Update timestamp
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    return ImageMetadataRecord(**data)


def _merge_dimension(existing: dict, update: dict) -> dict:
    """Merge two dimension records, preferring update for non-None fields."""
    merged = dict(existing)

    # Update top-level fields
    for key in ("label_source", "level_probs", "mos", "std"):
        if update.get(key) is not None:
            merged[key] = update[key]

    # Update nested optional fields
    for key in (
        "human",
        "siglip2",
        "cross_validation",
        "acceptance",
        "vlm_veto",
        "active_learning",
    ):
        if update.get(key) is not None:
            merged[key] = update[key]

    return merged
