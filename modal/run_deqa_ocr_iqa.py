"""Step 4: Run DeQA scoring on OCR-IQA correlation dataset via Modal.

Scores all 1,200 distorted images using 3 DeQA-Doc specialist models
(overall, sharpness, color) on a Modal L4 GPU. Extracts both MOS scores
and 5-level probability distributions.

Usage:
    # Upload images to Modal volume first
    uv run modal run modal/run_deqa_ocr_iqa.py --upload

    # Run inference (after upload)
    uv run modal run modal/run_deqa_ocr_iqa.py

    # Upload + run in one step
    uv run modal run modal/run_deqa_ocr_iqa.py --upload

    # Download results after completion
    uv run modal run modal/run_deqa_ocr_iqa.py --download-only

    # Detached (long-running)
    uv run modal run --detach modal/run_deqa_ocr_iqa.py
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("ocr-iqa-deqa-scoring")

# Persistent volumes
images_volume = modal.Volume.from_name(
    "ocr-iqa-images", create_if_missing=True
)
results_volume = modal.Volume.from_name(
    "ocr-iqa-deqa-results", create_if_missing=True
)
deqa_volume = modal.Volume.from_name("deqa-specialist-checkpoints")

# ============================================================================
# Modal Image (Python 3.10, torch 2.0.1 for mPLUG-Owl2)
# ============================================================================

_DEQA_ROOT = str(
    __import__("pathlib").Path(__file__).resolve().parent.parent / "DeQA-Score"
)

deqa_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "transformers==4.36.1",
        "peft==0.4.0",
        "accelerate==0.21.0",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "tqdm",
        "icecream",
        "sentencepiece",
        "modelscope",
        "bitsandbytes==0.43.3",
    )
    .add_local_dir(
        local_path=f"{_DEQA_ROOT}/src",
        remote_path="/root/deqa/src",
    )
    .add_local_dir(
        local_path=f"{_DEQA_ROOT}/preprocessor",
        remote_path="/root/deqa/preprocessor",
    )
)

# ============================================================================
# Constants
# ============================================================================

IMAGES_DIR = "/images"
RESULTS_DIR = "/results"
CHECKPOINT_BASE = f"{RESULTS_DIR}/checkpoints_ocr_iqa"

# DeQA level ordering: [excellent, good, fair, poor, bad] = [5, 4, 3, 2, 1]
LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]
MOS_WEIGHTS = [5.0, 4.0, 3.0, 2.0, 1.0]


# ============================================================================
# Checkpoint utilities (same pattern as benchmark_synthetic_ood.py)
# ============================================================================


def load_checkpoint(checkpoint_path: str) -> dict[str, dict]:
    """Load existing JSONL checkpoint for resume support."""
    import json
    from pathlib import Path

    results: dict[str, dict] = {}
    p = Path(checkpoint_path)
    if not p.exists():
        return results
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            key = f"{item['image_id']}_{item['tier']}"
            results[key] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def append_checkpoint(checkpoint_path: str, result: dict) -> None:
    """Append a single result to JSONL checkpoint."""
    import json
    from pathlib import Path

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result) + "\n")


# ============================================================================
# Volume upload (local entrypoint helper)
# ============================================================================


def upload_images_to_volume() -> int:
    """Upload distorted images + metadata to Modal volume."""
    import json
    from pathlib import Path

    data_dir = Path("research/ocr_iqa_correlation/data")
    distortion_meta = data_dir / "distortion_metadata.jsonl"

    if not distortion_meta.exists():
        raise FileNotFoundError(
            f"Distortion metadata not found: {distortion_meta}. Run step 02 first."
        )

    # Load metadata to get image paths
    records = []
    with open(distortion_meta) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Uploading {len(records)} images to Modal volume 'ocr-iqa-images'...")

    vol = modal.Volume.from_name("ocr-iqa-images", create_if_missing=True)

    tiers_uploaded: dict[str, int] = {}
    with vol.batch_upload(force=True) as batch:
        # Upload distortion metadata
        batch.put_file(str(distortion_meta), "distortion_metadata.jsonl")

        # Upload images by tier
        for record in records:
            image_path = record["image_path"]
            tier = record["tier"]
            image_id = record["image_id"]
            remote_path = f"{tier}/{image_id}.png"

            batch.put_file(image_path, remote_path)
            tiers_uploaded[tier] = tiers_uploaded.get(tier, 0) + 1

    for tier, count in sorted(tiers_uploaded.items()):
        print(f"  {tier}: {count} images")
    print(f"Total: {len(records)} images uploaded")
    return len(records)


# ============================================================================
# DeQA inference (runs on Modal GPU)
# ============================================================================


@app.function(
    image=deqa_image,
    gpu="L4",
    timeout=7200,
    volumes={
        IMAGES_DIR: images_volume,
        RESULTS_DIR: results_volume,
        "/deqa_checkpoints": deqa_volume,
    },
)
def run_deqa_scoring() -> list[dict]:
    """Score all OCR-IQA images with 3 DeQA-Doc specialists.

    Loads each specialist (overall, sharpness, color) sequentially,
    runs inference on all 1,200 images, extracts MOS + probability
    distributions, then frees GPU memory before loading next specialist.

    CRITICAL: Level ordering is [excellent,good,fair,poor,bad] = [5,4,3,2,1].
    MOS = np.inner(probs, [5,4,3,2,1]).
    """
    import gc
    import json
    import os
    import sys
    import time

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    # Add DeQA source to path
    sys.path.insert(0, "/root/deqa")

    from src.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from src.conversation import conv_templates
    from src.mm_utils import get_model_name_from_path, tokenizer_image_token
    from src.model.builder import load_pretrained_model

    # Load image records from volume
    meta_path = f"{IMAGES_DIR}/distortion_metadata.jsonl"
    records = []
    with open(meta_path) as f:
        for line in f:
            record = json.loads(line)
            # Remap paths to volume mount
            record["volume_path"] = (
                f"{IMAGES_DIR}/{record['tier']}/{record['image_id']}.png"
            )
            records.append(record)
    print(f"Loaded {len(records)} image records")

    # Load checkpoint
    ckpt_path = f"{CHECKPOINT_BASE}/deqa_ocr_iqa.jsonl"
    existing = load_checkpoint(ckpt_path)
    print(f"Existing checkpoint: {len(existing)} scored")

    if len(existing) >= len(records):
        print("All images already scored")
        results_volume.commit()
        return list(existing.values())

    # Level tokens and MOS weights
    mos_weights = np.array(MOS_WEIGHTS)

    # Find model path
    model_path = _resolve_deqa_model_path()

    # Per-image predictions: {key: {dim: {mos, probs}}}
    predictions: dict[str, dict] = {}

    dimensions = [
        ("overall", "overall"),
        ("sharpness", "sharpness"),
        ("color_fidelity", "color"),
    ]

    for dim_label, dim_dir_name in dimensions:
        print(f"\n{'=' * 50}")
        print(f"Loading specialist: {dim_label}")
        print(f"{'=' * 50}")

        specialist_path = f"{model_path}/deqa_0618_{dim_dir_name}_norm_pair_1024"
        if not os.path.isdir(specialist_path):
            specialist_path = model_path
            print(f"  Specialist not found, using shared: {specialist_path}")

        t0 = time.time()

        # Speed up loading
        torch.nn.Linear.reset_parameters = lambda self: None  # type: ignore[assignment]
        torch.nn.LayerNorm.reset_parameters = lambda self: None  # type: ignore[assignment]

        model_name = get_model_name_from_path(specialist_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            specialist_path,
            None,
            model_name,
            load_8bit=False,
            load_4bit=True,
            device="cuda",
            preprocessor_path="/root/deqa/preprocessor",
        )
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Get level token IDs
        level_ids = [tokenizer(tok)["input_ids"][1] for tok in LEVEL_NAMES]
        print(f"  Level IDs: {dict(zip(LEVEL_NAMES, level_ids))}")

        # Build prompt
        conv = conv_templates["mplug_owl2"].copy()
        inp = "How would you rate the quality of this image?"
        conv.append_message(conv.roles[0], inp + "\n" + DEFAULT_IMAGE_TOKEN)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + " The quality of the image is"

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )

        scored = 0
        skipped = 0
        for record in tqdm(records, desc=f"DeQA [{dim_label}]"):
            key = f"{record['image_id']}_{record['tier']}"

            # Skip if fully scored already
            if key in existing:
                skipped += 1
                continue

            if key not in predictions:
                predictions[key] = {"image_id": record["image_id"], "tier": record["tier"]}

            # Skip if this dimension already done for this image
            if dim_label in predictions[key]:
                continue

            try:
                pil_img = Image.open(record["volume_path"]).convert("RGB")

                # expand2square (matches DeQA preprocessing)
                w, h = pil_img.size
                if w != h:
                    bg = tuple(int(x * 255) for x in image_processor.image_mean)
                    side = max(w, h)
                    result_img = Image.new(pil_img.mode, (side, side), bg)
                    if w > h:
                        result_img.paste(pil_img, (0, (w - h) // 2))
                    else:
                        result_img.paste(pil_img, ((h - w) // 2, 0))
                    pil_img = result_img

                image_tensor = (
                    image_processor.preprocess(pil_img, return_tensors="pt")[
                        "pixel_values"
                    ]
                    .half()
                    .to("cuda")
                )

                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        images=image_tensor,
                    )
                    logits = output["logits"][:, -1, :]

                level_logits = logits[0, level_ids]
                probs = torch.softmax(level_logits, dim=0).cpu().numpy()
                mos = float(np.inner(probs, mos_weights))

                predictions[key][dim_label] = {
                    "mos": round(mos, 4),
                    "probs": [round(float(p), 4) for p in probs],
                }
                scored += 1

            except Exception as exc:
                print(f"  ERROR {key}: {exc}")
                predictions[key][dim_label] = {
                    "mos": float("nan"),
                    "probs": [0.0] * 5,
                    "error": str(exc),
                }

        print(f"  Scored: {scored}, Skipped (cached): {skipped}")

        # Free GPU memory
        del model, tokenizer, image_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("  GPU memory freed")

    # Write results checkpoint
    results = []
    for record in records:
        key = f"{record['image_id']}_{record['tier']}"

        if key in existing:
            results.append(existing[key])
            continue

        preds = predictions.get(key, {})
        result = {
            "image_id": record["image_id"],
            "tier": record["tier"],
        }

        has_all = all(
            dim_label in preds
            and not (isinstance(preds[dim_label], dict) and "error" in preds[dim_label])
            for dim_label, _ in dimensions
        )

        if has_all:
            result["deqa_overall_mos"] = preds["overall"]["mos"]
            result["deqa_overall_probs"] = preds["overall"]["probs"]
            result["deqa_sharpness_mos"] = preds["sharpness"]["mos"]
            result["deqa_sharpness_probs"] = preds["sharpness"]["probs"]
            result["deqa_color_mos"] = preds["color_fidelity"]["mos"]
            result["deqa_color_probs"] = preds["color_fidelity"]["probs"]
        else:
            result["error"] = "incomplete predictions"
            for dim_label, _ in dimensions:
                if dim_label in preds:
                    result[f"deqa_{dim_label}_mos"] = preds[dim_label].get("mos")

        results.append(result)
        append_checkpoint(ckpt_path, result)

    # Sanity check
    scored_results = [r for r in results if "deqa_overall_mos" in r]
    if scored_results:
        print(f"\nScored {len(scored_results)}/{len(results)} images")
        print("\nSanity check (first 5):")
        for r in scored_results[:5]:
            print(
                f"  {r['image_id']} [{r['tier']}]: "
                f"O={r['deqa_overall_mos']:.2f} "
                f"S={r['deqa_sharpness_mos']:.2f} "
                f"C={r['deqa_color_mos']:.2f}"
            )

        # Per-tier averages
        from collections import defaultdict
        tier_scores = defaultdict(list)
        for r in scored_results:
            tier_scores[r["tier"]].append(r["deqa_overall_mos"])
        print("\nPer-tier overall MOS averages:")
        for tier in ["ORIGINAL", "PRISTINE", "HIGH", "MEDIUM", "LOW", "DEGRADED"]:
            if tier in tier_scores:
                scores = tier_scores[tier]
                print(f"  {tier:12s}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    results_volume.commit()
    return results


def _resolve_deqa_model_path() -> str:
    """Find DeQA-Doc model path from checkpoint volume or ModelScope."""
    import os

    volume_base = "/deqa_checkpoints"
    expected_dirs = [
        "deqa_0618_overall_norm_pair_1024",
        "deqa_0618_sharpness_norm_pair_1024",
        "deqa_0618_color_norm_pair_1024",
    ]
    if all(os.path.isdir(f"{volume_base}/{d}") for d in expected_dirs):
        print(f"Found DeQA specialists on volume at {volume_base}")
        return volume_base

    print("Volume checkpoints not found, downloading from ModelScope: zhalala/DeQA-Doc")
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        local_path = snapshot_download(
            "zhalala/DeQA-Doc",
            cache_dir="/tmp/modelscope_cache",
        )
        print(f"Downloaded to: {local_path}")
        return local_path
    except Exception as exc:
        print(f"ModelScope download failed: {exc}")
        raise RuntimeError(
            "DeQA-Doc models not found. Upload to 'deqa-specialist-checkpoints' volume "
            "or ensure ModelScope access."
        ) from exc


# ============================================================================
# Local entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    upload: bool = False,
    download_only: bool = False,
):
    """Run DeQA scoring pipeline.

    Args:
        upload: Upload images to Modal volume before scoring.
        download_only: Just download existing results.
    """
    import json
    from pathlib import Path

    local_results_dir = Path("research/ocr_iqa_correlation/data/deqa_results")
    local_results_dir.mkdir(parents=True, exist_ok=True)
    local_results_path = local_results_dir / "deqa_scores.jsonl"

    if download_only:
        print("Downloading results from Modal volume...")
        vol = modal.Volume.from_name("ocr-iqa-deqa-results")
        ckpt_path = "checkpoints_ocr_iqa/deqa_ocr_iqa.jsonl"
        try:
            data = b""
            for chunk in vol.read_file(ckpt_path):
                data += chunk
            local_results_path.write_bytes(data)
            line_count = len(data.decode().strip().split("\n"))
            print(f"Downloaded {line_count} results to {local_results_path}")
        except Exception as e:
            print(f"Download failed: {e}")
        return

    if upload:
        count = upload_images_to_volume()
        print(f"Upload complete: {count} images")

    # Run scoring
    print("Starting DeQA scoring on Modal...")
    results = run_deqa_scoring.remote()

    # Save results locally
    with open(local_results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    scored = sum(1 for r in results if "deqa_overall_mos" in r)
    errors = sum(1 for r in results if "error" in r)
    print(f"\nResults saved to {local_results_path}")
    print(f"  Total: {len(results)}, Scored: {scored}, Errors: {errors}")
