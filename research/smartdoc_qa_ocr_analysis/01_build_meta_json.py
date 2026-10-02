"""Step 1: Build meta JSON files for DeQA inference on SmartDoc-QA images.

Generates the JSON manifest that iqa_eval.py expects: a list of dicts with
"image" keys pointing to relative paths from the root_dir.

Also generates a mapping file to join DeQA results back to OCR accuracy data.

Usage:
    cd DeQA-Score
    PYTHONPATH=./ .venv/bin/python \
        ../research/smartdoc_qa_ocr_analysis/01_build_meta_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET_ROOT = Path(
    "/mnt/e/image_detection/02_benchmark_only/smartdoc-qa/Dataset SmartDoc-QA"
)
OUTPUT_DIR = Path(__file__).parent / "data"


def build_meta() -> None:
    """Scan SmartDoc-QA images and build inference manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images_root = DATASET_ROOT / "Captured_Images"
    all_records: list[dict[str, str]] = []

    for phone_dir in sorted(images_root.iterdir()):
        if not phone_dir.is_dir():
            continue
        img_dir = phone_dir / "Images"
        if not img_dir.exists():
            continue

        phone_name = phone_dir.name  # Samsung_phone or Nokia_phone
        for img_file in sorted(img_dir.glob("*.jpg")):
            # Relative path from Captured_Images/
            rel_path = f"{phone_name}/Images/{img_file.name}"
            all_records.append({"image": rel_path})

    print(f"Total images found: {len(all_records)}")

    # Split into chunks of ~1000 for manageable inference batches
    chunk_size = 1000
    for i in range(0, len(all_records), chunk_size):
        chunk = all_records[i : i + chunk_size]
        chunk_idx = i // chunk_size
        out_path = OUTPUT_DIR / f"smartdoc_qa_meta_{chunk_idx:02d}.json"
        with open(out_path, "w") as f:
            json.dump(chunk, f, indent=2)
        print(f"  Wrote {len(chunk)} records to {out_path.name}")

    # Also write single combined file
    combined_path = OUTPUT_DIR / "smartdoc_qa_meta_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Wrote {len(all_records)} records to {combined_path.name}")


if __name__ == "__main__":
    build_meta()
