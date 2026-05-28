"""Parse SmartDoc-QA pre-computed OCR accuracy files and compute error rates.

Parses UNLV-ISRI accuracy report files (.cacc.txt, .wacc.txt) from the
SmartDoc-QA dataset and computes CER/WER statistics broken down by:
- OCR engine (Tesseract, FineReader)
- Phone device (Samsung, Nokia)
- Distortion type (Single vs Multiple)
- Document category (modern, administrative, receipts)
- Capture parameters (lighting, angle, blur type)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DATASET_ROOT = Path(
    "/mnt/e/image_detection/02_benchmark_only/smartdoc-qa/Dataset SmartDoc-QA"
)

# Document categories from README
DOC_CATEGORIES: dict[str, str] = {}
for d in range(1, 11):
    DOC_CATEGORIES[str(d)] = "modern"
for d in range(11, 21):
    DOC_CATEGORIES[str(d)] = "administrative"
for d in range(21, 31):
    DOC_CATEGORIES[str(d)] = "receipts"

PHONE_MAP = {
    "Samsung_phone": "Samsung",
    "Nokia_phone": "Nokia",
}

ENGINE_MAP = {
    "OCR_Accuracy_Tesseract": "Tesseract",
    "OCR_Accuracy_Finereader": "FineReader",
}


@dataclass
class AccuracyRecord:
    """Single accuracy measurement from one UNLV-ISRI report file."""

    filename: str
    phone: str
    engine: str
    document_id: int
    distortion_type: str  # "single" or "multiple"
    lighting: int
    angle_a: int
    angle_b: int
    blur_type: str  # "none", "Mb1", "Mb2", "Ob1", "Ob2", etc.
    category: str  # "modern", "administrative", "receipts"
    total_items: int  # characters or words
    errors: int
    accuracy_pct: float
    error_rate: float  # 1 - accuracy/100
    metric_type: str  # "CER" or "WER"


def parse_filename(fname: str) -> dict[str, Any]:
    """Extract capture parameters from SmartDoc-QA filename.

    Args:
        fname: Filename like S_Img_Android_D1_L1_r35_a0_b0 or
               M_Img_WP_D10_L2_r35_a-10_b5_Mb2

    Returns:
        Dictionary with parsed parameters.
    """
    # Remove extension parts
    base = fname
    for suffix in [".cacc.txt", ".wacc.txt", ".jpg", ".txt"]:
        base = base.removesuffix(suffix)

    result: dict[str, Any] = {"raw": base}

    # Distortion type
    result["distortion_type"] = "single" if base.startswith("S_") else "multiple"

    # Document number
    d_match = re.search(r"_D(\d+)_", base)
    result["document_id"] = int(d_match.group(1)) if d_match else -1

    # Lighting
    l_match = re.search(r"_L(\d+)_", base)
    result["lighting"] = int(l_match.group(1)) if l_match else -1

    # Angles
    a_match = re.search(r"_a(-?\d+)_", base)
    result["angle_a"] = int(a_match.group(1)) if a_match else 0

    b_match = re.search(r"_b(-?\d+)", base)
    result["angle_b"] = int(b_match.group(1)) if b_match else 0

    # Blur type - check for Mb or Ob suffix
    blur_match = re.search(r"_(Mb\d+|Ob\d+)$", base)
    result["blur_type"] = blur_match.group(1) if blur_match else "none"

    return result


def parse_cacc_file(filepath: Path) -> tuple[int, int, float] | None:
    """Parse UNLV-ISRI character accuracy report.

    Args:
        filepath: Path to .cacc.txt file.

    Returns:
        Tuple of (total_chars, errors, accuracy_pct) or None if parse fails.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    chars_match = re.search(r"(\d+)\s+Characters\b", text)
    errors_match = re.search(r"(\d+)\s+Errors\b", text)
    acc_match = re.search(r"([\d.]+)%\s+Accuracy\b", text)

    if not (chars_match and errors_match and acc_match):
        return None

    return (
        int(chars_match.group(1)),
        int(errors_match.group(1)),
        float(acc_match.group(1)),
    )


def parse_wacc_file(filepath: Path) -> tuple[int, int, float] | None:
    """Parse UNLV-ISRI word accuracy report.

    Args:
        filepath: Path to .wacc.txt file.

    Returns:
        Tuple of (total_words, misrecognized, accuracy_pct) or None if parse fails.
    """
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    words_match = re.search(r"(\d+)\s+Words\b", text)
    missed_match = re.search(r"(\d+)\s+Misrecognized\b", text)
    acc_match = re.search(r"([\d.]+)%\s+Accuracy\b", text)

    if not (words_match and missed_match and acc_match):
        return None

    return (
        int(words_match.group(1)),
        int(missed_match.group(1)),
        float(acc_match.group(1)),
    )


def collect_all_records() -> list[AccuracyRecord]:
    """Walk the dataset and parse all accuracy files.

    Returns:
        List of AccuracyRecord objects.
    """
    records: list[AccuracyRecord] = []

    for phone_dir_name, phone_label in PHONE_MAP.items():
        phone_dir = DATASET_ROOT / "Captured_Images" / phone_dir_name

        for engine_dir_name, engine_label in ENGINE_MAP.items():
            acc_dir = phone_dir / engine_dir_name

            if not acc_dir.exists():
                print(f"WARNING: {acc_dir} not found, skipping")
                continue

            # Process character accuracy files
            for cacc_file in sorted(acc_dir.glob("*.cacc.txt")):
                parsed = parse_cacc_file(cacc_file)
                if parsed is None:
                    continue

                total, errors, acc_pct = parsed
                params = parse_filename(cacc_file.name)
                doc_id = params["document_id"]

                records.append(
                    AccuracyRecord(
                        filename=cacc_file.stem.removesuffix(".cacc"),
                        phone=phone_label,
                        engine=engine_label,
                        document_id=doc_id,
                        distortion_type=params["distortion_type"],
                        lighting=params["lighting"],
                        angle_a=params["angle_a"],
                        angle_b=params["angle_b"],
                        blur_type=params["blur_type"],
                        category=DOC_CATEGORIES.get(str(doc_id), "unknown"),
                        total_items=total,
                        errors=errors,
                        accuracy_pct=acc_pct,
                        error_rate=1.0 - acc_pct / 100.0,
                        metric_type="CER",
                    )
                )

            # Process word accuracy files
            for wacc_file in sorted(acc_dir.glob("*.wacc.txt")):
                parsed = parse_wacc_file(wacc_file)
                if parsed is None:
                    continue

                total, errors, acc_pct = parsed
                params = parse_filename(wacc_file.name)
                doc_id = params["document_id"]

                records.append(
                    AccuracyRecord(
                        filename=wacc_file.stem.removesuffix(".wacc"),
                        phone=phone_label,
                        engine=engine_label,
                        document_id=doc_id,
                        distortion_type=params["distortion_type"],
                        lighting=params["lighting"],
                        angle_a=params["angle_a"],
                        angle_b=params["angle_b"],
                        blur_type=params["blur_type"],
                        category=DOC_CATEGORIES.get(str(doc_id), "unknown"),
                        total_items=total,
                        errors=errors,
                        accuracy_pct=acc_pct,
                        error_rate=1.0 - acc_pct / 100.0,
                        metric_type="WER",
                    )
                )

    return records


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def print_stats_table(
    title: str,
    groups: dict[str, list[float]],
    metric_name: str = "Error Rate",
) -> None:
    """Print a formatted statistics table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(
        f"  {'Group':<25} {'Count':>6} {'Mean':>8} {'Std':>8} "
        f"{'Median':>8} {'Min':>8} {'Max':>8}"
    )
    print(f"  {'-' * 25} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for group_name in sorted(groups.keys()):
        values = groups[group_name]
        if not values:
            continue
        s = compute_stats(values)
        print(
            f"  {group_name:<25} {s['count']:>6} {s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['median']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}"
        )


def main() -> None:
    """Parse SmartDoc-QA accuracy files and print error rate analysis."""
    print("Parsing SmartDoc-QA OCR accuracy files...")
    records = collect_all_records()

    cer_records = [r for r in records if r.metric_type == "CER"]
    wer_records = [r for r in records if r.metric_type == "WER"]

    print(f"\nTotal records: {len(records)}")
    print(f"  CER records: {len(cer_records)}")
    print(f"  WER records: {len(wer_records)}")

    # --- Overall CER/WER ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        error_rates = [r.error_rate for r in recs]
        s = compute_stats(error_rates)
        print(f"\nOverall {metric}: mean={s['mean']:.4f}, "
              f"median={s['median']:.4f}, std={s['std']:.4f}")

    # --- By Engine ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in recs:
            groups[r.engine].append(r.error_rate)
        print_stats_table(f"{metric} by OCR Engine", groups)

    # --- By Phone ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[r.phone].append(r.error_rate)
        print_stats_table(f"{metric} by Phone Device", groups)

    # --- By Distortion Type ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[r.distortion_type].append(r.error_rate)
        print_stats_table(f"{metric} by Distortion Type (Single vs Multiple)", groups)

    # --- By Document Category ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[r.category].append(r.error_rate)
        print_stats_table(f"{metric} by Document Category", groups)

    # --- By Lighting Condition ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[f"L{r.lighting}"].append(r.error_rate)
        print_stats_table(f"{metric} by Lighting Condition", groups)

    # --- By Blur Type ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[r.blur_type].append(r.error_rate)
        print_stats_table(f"{metric} by Blur Type", groups)

    # --- By Engine x Distortion Type ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[f"{r.engine} / {r.distortion_type}"].append(r.error_rate)
        print_stats_table(f"{metric} by Engine x Distortion Type", groups)

    # --- By Engine x Phone ---
    for metric, recs in [("CER", cer_records), ("WER", wer_records)]:
        groups = defaultdict(list)
        for r in recs:
            groups[f"{r.engine} / {r.phone}"].append(r.error_rate)
        print_stats_table(f"{metric} by Engine x Phone", groups)

    # --- Export to JSONL ---
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "smartdoc_qa_error_rates.jsonl"

    with jsonl_path.open("w") as f:
        for r in records:
            row = {
                "filename": r.filename,
                "phone": r.phone,
                "engine": r.engine,
                "document_id": r.document_id,
                "distortion_type": r.distortion_type,
                "lighting": r.lighting,
                "angle_a": r.angle_a,
                "angle_b": r.angle_b,
                "blur_type": r.blur_type,
                "category": r.category,
                "total_items": r.total_items,
                "errors": r.errors,
                "accuracy_pct": r.accuracy_pct,
                "error_rate": round(r.error_rate, 6),
                "metric_type": r.metric_type,
            }
            f.write(json.dumps(row) + "\n")

    print(f"\nExported {len(records)} records to {jsonl_path}")


if __name__ == "__main__":
    main()
