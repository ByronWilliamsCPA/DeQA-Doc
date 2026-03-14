"""Run VLM-based OCR models on OCR-IQA dataset via Modal.

Runs DeepSeek-OCR2, PaddleOCR-VL-1.5, and GLM-OCR on all 1,200
distorted document images using HuggingFace transformers.
Results are saved as JSONL per-engine files compatible with the
existing OCR analysis pipeline.

Each model requires different transformers versions and APIs:
  - DeepSeek-OCR2: transformers<5.0, AutoModel + model.infer()
  - PaddleOCR-VL-1.5: transformers>=5.0, AutoModelForImageTextToText
  - GLM-OCR: transformers>=5.0, AutoModelForImageTextToText

Usage:
    # Run single model
    uv run modal run modal/run_vlm_ocr.py --model deepseek-ocr2

    # Run all models
    uv run modal run modal/run_vlm_ocr.py

    # Download results
    uv run modal run modal/run_vlm_ocr.py --download-only

    # Detached (long-running)
    uv run modal run --detach modal/run_vlm_ocr.py --model deepseek-ocr2
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("ocr-iqa-vlm-ocr")

images_volume = modal.Volume.from_name("ocr-iqa-images", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "ocr-iqa-vlm-ocr-results", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODELS = {
    "deepseek-ocr2": {
        "hf_id": "deepseek-ai/DeepSeek-OCR-2",
        "gpu": "A10G",
    },
    "paddleocr-vl": {
        "hf_id": "PaddlePaddle/PaddleOCR-VL-1.5",
        "gpu": "L4",
    },
    "glm-ocr": {
        "hf_id": "zai-org/GLM-OCR",
        "gpu": "L4",
    },
    "mineru2.5": {
        "hf_id": "opendatalab/MinerU2.5-2509-1.2B",
        "gpu": "L4",
    },
}

IMAGES_DIR = "/images"
RESULTS_DIR = "/results"

# ============================================================================
# Modal Images — separate for transformers 4.x and 5.x
# ============================================================================

# DeepSeek-OCR-2 needs transformers==4.46.3 (model card requirement)
# Also needs easydict for its custom model code
image_tf4 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers==4.46.3",
        "accelerate>=0.30.0",
        "Pillow",
        "tqdm",
        "sentencepiece",
        "protobuf",
        "addict",
        "matplotlib",
        "timm",
        "einops",
        "easydict",
    )
    .env({"ATTN_IMPLEMENTATION": "eager"})
)

# PaddleOCR-VL-1.5 and GLM-OCR work with transformers>=5.0
image_tf5 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers>=5.0.0",
        "accelerate>=0.30.0",
        "Pillow",
        "tqdm",
        "sentencepiece",
        "protobuf",
    )
    .env({"ATTN_IMPLEMENTATION": "eager"})
)

# MinerU2.5 needs mineru-vl-utils which pins transformers>=4.51.1,<5.0
image_mineru = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "accelerate>=0.30.0",
        "Pillow",
        "tqdm",
        "mineru-vl-utils[transformers]",
    )
    .env({"ATTN_IMPLEMENTATION": "eager"})
)

VOLUME_MOUNTS = {
    IMAGES_DIR: images_volume,
    RESULTS_DIR: results_volume,
    "/hf_cache": hf_cache_volume,
}


# ============================================================================
# Checkpoint utilities
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


def load_image_records() -> list[dict]:
    """Load image metadata records from volume."""
    import json

    meta_path = f"{IMAGES_DIR}/distortion_metadata.jsonl"
    records = []
    with open(meta_path) as f:
        for line in f:
            record = json.loads(line)
            record["volume_path"] = (
                f"{IMAGES_DIR}/{record['tier']}/{record['image_id']}.png"
            )
            records.append(record)
    return records


def process_images(model_key: str, records: list[dict], infer_fn) -> list[dict]:
    """Common loop: checkpoint, infer, save results.

    Args:
        model_key: Engine name for checkpointing.
        records: Image metadata records.
        infer_fn: Callable(image_path) -> str, returns OCR text.

    Returns:
        List of all result dicts.
    """
    import time

    ckpt_path = f"{RESULTS_DIR}/checkpoints/{model_key}.jsonl"
    existing = load_checkpoint(ckpt_path)
    print(f"Existing checkpoint: {len(existing)} processed")

    remaining = [
        r for r in records if f"{r['image_id']}_{r['tier']}" not in existing
    ]
    print(f"Remaining: {len(remaining)} images")

    if not remaining:
        print("All images already processed")
        results_volume.commit()
        return list(existing.values())

    all_results = list(existing.values())

    for idx, record in enumerate(remaining):
        t0 = time.time()
        try:
            text = infer_fn(record["volume_path"])
            elapsed_ms = (time.time() - t0) * 1000
            result = {
                "image_id": record["image_id"],
                "tier": record["tier"],
                "engine": model_key,
                "ocr_text": text,
                "ocr_chars": len(text),
                "time_ms": round(elapsed_ms, 1),
                "error": None,
            }
            if (idx + 1) % 50 == 0:
                print(
                    f"  [{idx + 1}/{len(remaining)}] "
                    f"{record['image_id']}/{record['tier']} "
                    f"chars={len(text)} ({elapsed_ms:.0f}ms)"
                )
        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000
            print(f"  ERROR [{idx + 1}] {record['image_id']}: {exc}")
            result = {
                "image_id": record["image_id"],
                "tier": record["tier"],
                "engine": model_key,
                "ocr_text": "",
                "ocr_chars": 0,
                "time_ms": round(elapsed_ms, 1),
                "error": str(exc),
            }

        all_results.append(result)
        append_checkpoint(ckpt_path, result)

        # Commit every 100 images for durability
        if (idx + 1) % 100 == 0:
            results_volume.commit()

    ok = sum(1 for r in all_results if not r.get("error"))
    err = sum(1 for r in all_results if r.get("error"))
    print(f"\nComplete: {ok} success, {err} errors out of {len(all_results)}")
    results_volume.commit()
    return all_results


# ============================================================================
# DeepSeek-OCR-2 (transformers 4.x, custom model.infer() API)
# ============================================================================


@app.function(
    image=image_tf4,
    gpu="A10G",
    timeout=21600,
    volumes=VOLUME_MOUNTS,
)
def run_deepseek_ocr2() -> list[dict]:
    """Run DeepSeek-OCR-2 on all images."""
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    records = load_image_records()
    print(f"Loaded {len(records)} image records")

    hf_id = "deepseek-ai/DeepSeek-OCR-2"
    print(f"Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=True, cache_dir="/hf_cache"
    )
    model = AutoModel.from_pretrained(
        hf_id,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="eager",
        cache_dir="/hf_cache",
    )
    model = model.eval().cuda().to(torch.bfloat16)
    print("Model loaded")

    # DeepSeek-OCR-2 uses a custom infer() method that writes to file.
    # It returns None — output must be read from {output_path}/result.mmd.
    prompt = "<image>\nFree OCR. "
    tmp_output = "/tmp/deepseek_ocr_output"
    result_file = f"{tmp_output}/result.mmd"
    os.makedirs(tmp_output, exist_ok=True)

    def infer_fn(image_path: str) -> str:
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=tmp_output,
            save_results=True,
        )
        with open(result_file) as f:
            return f.read().strip()

    return process_images("deepseek-ocr2", records, infer_fn)


# ============================================================================
# PaddleOCR-VL-1.5 (transformers 5.x)
# ============================================================================


@app.function(
    image=image_tf5,
    gpu="L4",
    timeout=21600,
    volumes=VOLUME_MOUNTS,
)
def run_paddleocr_vl() -> list[dict]:
    """Run PaddleOCR-VL-1.5 on all images."""
    import os

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    records = load_image_records()
    print(f"Loaded {len(records)} image records")

    hf_id = "PaddlePaddle/PaddleOCR-VL-1.5"
    print(f"Loading {hf_id}...")
    processor = AutoProcessor.from_pretrained(hf_id, cache_dir="/hf_cache")
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        cache_dir="/hf_cache",
    ).cuda().eval()
    print("Model loaded")

    # Image size constraints per model card
    max_pixels = 1280 * 28 * 28

    def infer_fn(image_path: str) -> str:
        pil_img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        return text

    return process_images("paddleocr-vl", records, infer_fn)


# ============================================================================
# GLM-OCR (transformers 5.x)
# ============================================================================


@app.function(
    image=image_tf5,
    gpu="L4",
    timeout=21600,
    volumes=VOLUME_MOUNTS,
)
def run_glm_ocr() -> list[dict]:
    """Run GLM-OCR on all images."""
    import os

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    records = load_image_records()
    print(f"Loaded {len(records)} image records")

    hf_id = "zai-org/GLM-OCR"
    print(f"Loading {hf_id}...")
    processor = AutoProcessor.from_pretrained(hf_id, cache_dir="/hf_cache")
    model = AutoModelForImageTextToText.from_pretrained(
        hf_id,
        torch_dtype="auto",
        device_map="auto",
        cache_dir="/hf_cache",
    )
    print("Model loaded")

    def infer_fn(image_path: str) -> str:
        pil_img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        text = processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        return text

    return process_images("glm-ocr", records, infer_fn)


# ============================================================================
# MinerU2.5 (transformers 5.x, Qwen2VL-based, two-step extraction)
# ============================================================================


@app.function(
    image=image_mineru,
    gpu="L4",
    timeout=21600,
    volumes=VOLUME_MOUNTS,
)
def run_mineru25() -> list[dict]:
    """Run MinerU2.5 on all images."""
    import os

    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache"

    records = load_image_records()
    print(f"Loaded {len(records)} image records")

    hf_id = "opendatalab/MinerU2.5-2509-1.2B"
    print(f"Loading {hf_id}...")

    from mineru_vl_utils import MinerUClient

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        hf_id,
        torch_dtype="auto",
        device_map="auto",
        cache_dir="/hf_cache",
    )
    processor = AutoProcessor.from_pretrained(
        hf_id, use_fast=True, cache_dir="/hf_cache"
    )
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
    )
    print("Model loaded")

    def infer_fn(image_path: str) -> str:
        pil_img = Image.open(image_path).convert("RGB")
        blocks = client.two_step_extract(pil_img)
        # Concatenate text from all extracted blocks
        text_parts = []
        for block in blocks:
            if isinstance(block, dict):
                text_parts.append(block.get("text", str(block)))
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)

    return process_images("mineru2.5", records, infer_fn)


# ============================================================================
# Dispatch map
# ============================================================================

MODEL_FUNCTIONS = {
    "deepseek-ocr2": run_deepseek_ocr2,
    "paddleocr-vl": run_paddleocr_vl,
    "glm-ocr": run_glm_ocr,
    "mineru2.5": run_mineru25,
}


# ============================================================================
# Download results
# ============================================================================


def download_results() -> None:
    """Download all VLM OCR results from Modal volume."""
    from pathlib import Path

    local_dir = Path("research/ocr_iqa_correlation/data/ocr_results")
    local_dir.mkdir(parents=True, exist_ok=True)

    vol = modal.Volume.from_name("ocr-iqa-vlm-ocr-results")

    for model_key in MODELS:
        remote_path = f"checkpoints/{model_key}.jsonl"
        local_path = local_dir / f"{model_key}.jsonl"

        try:
            data = b""
            for chunk in vol.read_file(remote_path):
                data += chunk
            local_path.write_bytes(data)
            line_count = len(data.decode().strip().split("\n"))
            print(f"  {model_key}: {line_count} results -> {local_path}")
        except Exception as e:
            print(f"  {model_key}: download failed: {e}")


# ============================================================================
# Local entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    download_only: bool = False,
    model: str | None = None,
):
    """Run VLM OCR pipeline.

    Args:
        download_only: Just download existing results.
        model: Run only this model (e.g., 'deepseek-ocr2').
    """
    import json
    from pathlib import Path

    if download_only:
        print("Downloading VLM OCR results from Modal volume...")
        download_results()
        return

    models_to_run = list(MODELS.keys())
    if model:
        if model not in MODELS:
            print(
                f"ERROR: Unknown model '{model}'. "
                f"Available: {list(MODELS.keys())}"
            )
            return
        models_to_run = [model]

    local_dir = Path("research/ocr_iqa_correlation/data/ocr_results")
    local_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running: {model_key} ({MODELS[model_key]['hf_id']})")
        print(f"{'=' * 60}")

        fn = MODEL_FUNCTIONS[model_key]
        results = fn.remote()

        # Save locally
        local_path = local_dir / f"{model_key}.jsonl"
        with open(local_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        ok = sum(1 for r in results if not r.get("error"))
        err = sum(1 for r in results if r.get("error"))
        print(f"  Saved: {local_path} ({ok} success, {err} errors)")

    print("\nAll VLM OCR models complete.")
