"""Benchmark fine-tuned IQA models on 520-image synthetic OOD dataset.

Evaluates 3 models (SigLIP2-IQA, HyperIQA++, DeQA-Doc specialists) on
synthetic OOD documents and compares against VLM teacher baselines.

Usage:
    # All models
    uv run modal run modal/benchmark_synthetic_ood.py

    # Single model
    uv run modal run modal/benchmark_synthetic_ood.py --model siglip2
    uv run modal run modal/benchmark_synthetic_ood.py --model hyperiqa
    uv run modal run modal/benchmark_synthetic_ood.py --model deqa

    # Metrics only (from existing checkpoints)
    uv run modal run modal/benchmark_synthetic_ood.py --metrics-only

    # Detached (long-running)
    uv run modal run --detach modal/benchmark_synthetic_ood.py
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("synthetic-ood-benchmark")

# Persistent volumes
synthetic_volume = modal.Volume.from_name(
    "synthetic-ood-data", create_if_missing=True
)
siglip2_volume = modal.Volume.from_name("siglip2-iqa-results")
hyperiqa_volume = modal.Volume.from_name("hyperiqa-checkpoints")
results_volume = modal.Volume.from_name(
    "synthetic-ood-results", create_if_missing=True
)
deqa_volume = modal.Volume.from_name("deqa-specialist-checkpoints")

# ============================================================================
# Modal Images (two separate images for torch version compatibility)
# ============================================================================

# SigLIP2: needs transformers>=4.51 for SigLIP2 model support
siglip2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "tqdm",
    )
)

# HyperIQA++: needs pyiqa which pins transformers==4.37.2
# pyiqa pulls opencv-python which needs libGL + libgthread
hyperiqa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pyiqa>=0.1.12",
        "timm>=1.0.0",
        "scipy",
        "numpy<2.0",
        "Pillow",
        "tqdm",
    )
)

# Alias for metrics computation (lightweight, no GPU)
modern_image = siglip2_image

# DeQA-Doc: legacy torch for mPLUG-Owl2 compatibility
# Resolve paths relative to this script's location (DeQA-Doc repo root)
_DEQA_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "DeQA-Score")

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
# Shared utilities (inlined, no cross-repo dependency)
# ============================================================================

SYNTHETIC_DATA_DIR = "/synthetic/ood_poc_test"
CHECKPOINT_BASE = "/results/checkpoints_synthetic"

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42

# wSRCC weights (VQualA 2025 competition metric)
W_OVERALL = 0.5
W_SHARPNESS = 0.25
W_COLOR = 0.25


def load_synthetic_dataset(data_dir: str = SYNTHETIC_DATA_DIR) -> list[dict]:
    """Load synthetic dataset metadata from JSONL."""
    import json
    from pathlib import Path

    meta_path = Path(data_dir) / "metadata.jsonl"
    images = []
    with meta_path.open() as f:
        for line in f:
            d = json.loads(line)
            # Remap image path to volume mount
            d["image_path"] = str(Path(data_dir) / d["image_id"])
            images.append(d)
    return images


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
            results[item["image_id"]] = item
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
# SigLIP2 Model (inlined from image_detection)
# ============================================================================


def _build_siglip2_model(model_id: str = "google/siglip2-base-patch16-naflex"):
    """Build multi-task SigLIP2 model architecture.

    Inlined from image_detection/src/.../siglip2_multitask.py:_build_model.
    Only IQA heads are needed for this benchmark, but we load all heads
    to match the checkpoint's state_dict.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    # Constants matching training config
    n_script = 19
    n_source = 3
    n_orientation = 4

    head_configs = {
        "overall": {"hidden": 256, "out": 2, "drop": 0.3, "type": "reg"},
        "sharpness": {"hidden": 256, "out": 2, "drop": 0.3, "type": "reg"},
        "color": {"hidden": 256, "out": 2, "drop": 0.3, "type": "reg"},
        "script": {"hidden": 256, "out": n_script, "drop": 0.3, "type": "cls"},
        "source": {"hidden": 64, "out": n_source, "drop": 0.0, "type": "cls"},
        "orientation": {"hidden": 64, "out": n_orientation, "drop": 0.0, "type": "cls"},
        "shadow": {"hidden": 64, "out": 2, "drop": 0.0, "type": "reg"},
        "warping": {"hidden": 64, "out": 2, "drop": 0.0, "type": "reg"},
    }

    backbone = AutoModel.from_pretrained(model_id)
    embed_dim = backbone.config.vision_config.hidden_size

    heads = nn.ModuleDict()
    head_types: dict[str, str] = {}

    for name, cfg in head_configs.items():
        layers: list[nn.Module] = [nn.Linear(embed_dim, cfg["hidden"]), nn.ReLU()]
        if cfg["drop"] > 0:
            layers.append(nn.Dropout(cfg["drop"]))
        layers.append(nn.Linear(cfg["hidden"], cfg["out"]))
        heads[name] = nn.Sequential(*layers)
        head_types[name] = cfg["type"]

    class _MultiTaskModel(nn.Module):
        def __init__(self, bb, hds, htypes):
            super().__init__()
            self.backbone = bb
            self.heads = hds
            self._head_types = htypes
            for hname, hcfg in head_configs.items():
                if hcfg["type"] == "reg":
                    self.register_buffer(f"temp_{hname}", torch.tensor(1.0))

        def forward(self, pixel_values, spatial_shapes=None, tasks=None):
            out = self.backbone.get_image_features(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
            )
            # Newer transformers returns BaseModelOutputWithPooling, not a tensor
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                features = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                # Use mean pooling over sequence dim
                features = out.last_hidden_state.mean(dim=1)
            else:
                features = out
            active = list(self.heads.keys()) if tasks is None else tasks
            results = {}
            for task_name in active:
                if task_name not in self.heads:
                    continue
                out = self.heads[task_name](features)
                if self._head_types.get(task_name) == "reg":
                    mu = out[:, 0]
                    log_sigma_sq = out[:, 1]
                    sigma_sq = torch.exp(log_sigma_sq)
                    temp = getattr(self, f"temp_{task_name}")
                    results[task_name] = {"mu": mu, "sigma_sq": temp * sigma_sq}
                else:
                    results[task_name] = out
            return results

    return _MultiTaskModel(backbone, heads, head_types)


@app.function(
    image=siglip2_image,
    gpu="L4",
    timeout=1800,
    volumes={
        "/synthetic": synthetic_volume,
        "/siglip2_vol": siglip2_volume,
        "/results": results_volume,
    },
)
def evaluate_siglip2() -> list[dict]:
    """Evaluate SigLIP2-IQA-Base-86M on synthetic OOD dataset."""
    import time

    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor

    dataset = load_synthetic_dataset()
    ckpt_path = f"{CHECKPOINT_BASE}/siglip2_iqa_base_86m.jsonl"
    existing = load_checkpoint(ckpt_path)
    print(f"SigLIP2: {len(dataset)} images, {len(existing)} cached")

    # Load model
    t0 = time.time()
    model_id = "google/siglip2-base-patch16-naflex"
    model = _build_siglip2_model(model_id)

    checkpoint_file = "/siglip2_vol/siglip2/siglip2_iqa_best.pt"
    ckpt = torch.load(checkpoint_file, map_location="cuda", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    results = []
    for item in tqdm(dataset, desc="SigLIP2"):
        img_id = item["image_id"]
        if img_id in existing:
            results.append(existing[img_id])
            continue

        try:
            pil_img = Image.open(item["image_path"]).convert("RGB")
            inputs = processor(
                images=pil_img,
                return_tensors="pt",
                max_num_patches=784,
                padding="max_length",
            )
            pixel_values = inputs["pixel_values"].to("cuda")
            spatial_shapes = inputs["spatial_shapes"].to("cuda")

            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    spatial_shapes=spatial_shapes,
                    tasks=["overall", "sharpness", "color"],
                )

            # Rescale from [0,1] to [1,5] MOS
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": {
                    "overall": outputs["overall"]["mu"][0].item() * 4.0 + 1.0,
                    "sharpness": outputs["sharpness"]["mu"][0].item() * 4.0 + 1.0,
                    "color": outputs["color"]["mu"][0].item() * 4.0 + 1.0,
                },
                "ground_truth": item["synthetic_scores"],
            }
        except Exception as exc:
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": None,
                "ground_truth": item["synthetic_scores"],
                "error": str(exc),
            }

        results.append(result)
        append_checkpoint(ckpt_path, result)

    # Sanity check: print first 5 predictions
    print("\nSanity check (first 5 predictions):")
    for r in results[:5]:
        if r.get("predicted"):
            p = r["predicted"]
            g = r["ground_truth"]
            print(
                f"  {r['image_id']}: pred O={p['overall']:.2f} S={p['sharpness']:.2f} "
                f"C={p['color']:.2f} | gt O={g['overall']:.2f} S={g['sharpness']:.2f} "
                f"C={g['color']:.2f}"
            )

    results_volume.commit()
    return results


# ============================================================================
# HyperIQA++ Model (inlined from git commit 9e63791)
# ============================================================================


def _build_hyperiqa_model(checkpoint_path: str):
    """Build and load HyperIQA++ model.

    Architecture inlined from image_detection git commit 9e63791:
    - src/.../hyperiqa_plus_plus/model.py
    - src/.../hyperiqa_plus_plus/modules.py
    """
    from typing import cast

    import pyiqa
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812
    from torch import Tensor

    # --- Inlined modules ---

    class SpatialAttentionModule(nn.Module):
        def __init__(self, in_channels: int = 2048):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(512)
            self.conv2 = nn.Conv2d(512, 1, kernel_size=1)

        def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
            attn = F.relu(self.bn1(self.conv1(features)))
            attn = torch.sigmoid(self.conv2(attn))
            return features * attn, attn

    class SoftLabelHead(nn.Module):
        def __init__(self, embed_dim: int = 2048, num_bins: int = 10):
            super().__init__()
            self.num_bins = num_bins
            self.head = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_bins),
            )
            self.register_buffer("bin_centers", torch.linspace(1.0, 5.0, num_bins))

        def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            logits = cast(Tensor, self.head(features))
            probs = F.softmax(logits, dim=-1)
            bin_centers = cast(Tensor, self.bin_centers)
            score = (probs * bin_centers).sum(dim=-1)
            return score, probs, logits

    class HyperIQAPlusPlus(nn.Module):
        def __init__(self, num_bins: int = 10):
            super().__init__()
            metric = pyiqa.create_metric("hyperiqa", device="cpu", as_loss=True)
            hyperiqa_model = metric.net

            if hasattr(hyperiqa_model, "res"):
                self.backbone = hyperiqa_model.res
            elif hasattr(hyperiqa_model, "backbone"):
                self.backbone = hyperiqa_model.backbone
            else:
                self.backbone = hyperiqa_model

            if hasattr(hyperiqa_model, "hypernet"):
                self.hypernet = hyperiqa_model.hypernet
            else:
                self.hypernet = nn.Sequential(
                    nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.2),
                )

            self.feature_fusion = None
            self.use_multiscale_fusion = False
            self.spatial_attention = SpatialAttentionModule(in_channels=2048)
            self.head_overall = SoftLabelHead(embed_dim=2048, num_bins=num_bins)
            self.head_sharpness = SoftLabelHead(embed_dim=2048, num_bins=num_bins)
            self.head_color = SoftLabelHead(embed_dim=2048, num_bins=num_bins)

        def forward(self, x: Tensor) -> dict:
            if hasattr(self.backbone, "base_model"):
                features = self.backbone.base_model(x)
                fused = features[-1] if isinstance(features, list) else features
            else:
                features = self.backbone(x)
                if isinstance(features, list):
                    fused = features[-1]
                elif len(features.shape) == 2:
                    fused = features.view(features.size(0), -1, 1, 1)
                else:
                    fused = features

            if len(fused.shape) == 4:
                attended, attn_map = self.spatial_attention(fused)
                feat = F.adaptive_avg_pool2d(attended, (1, 1)).flatten(1)
            else:
                feat = fused
                attn_map = None

            if self.hypernet is not None:
                feat = self.hypernet(feat)

            o_score, o_probs, o_logits = self.head_overall(feat)
            s_score, s_probs, s_logits = self.head_sharpness(feat)
            c_score, c_probs, c_logits = self.head_color(feat)

            return {
                "overall": {"score": o_score, "probs": o_probs, "logits": o_logits},
                "sharpness": {"score": s_score, "probs": s_probs, "logits": s_logits},
                "color": {"score": c_score, "probs": c_probs, "logits": c_logits},
                "attention_map": attn_map,
            }

    # Build and load
    model = HyperIQAPlusPlus(num_bins=10)

    # Stub out training module that checkpoint pickle references
    import sys
    import types

    class _TrainingConfigStub:
        """Dummy class to unpickle checkpoint config."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def __reduce__(self):
            return (self.__class__, (), self.__dict__)
        def __setstate__(self, state):
            self.__dict__.update(state)

    stub_mod = types.ModuleType("train_hyperiqa_plus_plus")
    stub_mod.TrainingConfig = _TrainingConfigStub  # type: ignore[attr-defined]
    sys.modules["train_hyperiqa_plus_plus"] = stub_mod

    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"HyperIQA++ loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model


@app.function(
    image=hyperiqa_image,
    gpu="L4",
    timeout=1800,
    volumes={
        "/synthetic": synthetic_volume,
        "/hyperiqa_vol": hyperiqa_volume,
        "/results": results_volume,
    },
)
def evaluate_hyperiqa() -> list[dict]:
    """Evaluate HyperIQA++-DIQA5000 on synthetic OOD dataset."""
    import time

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    dataset = load_synthetic_dataset()
    ckpt_path = f"{CHECKPOINT_BASE}/hyperiqa_plus_plus.jsonl"
    existing = load_checkpoint(ckpt_path)
    print(f"HyperIQA++: {len(dataset)} images, {len(existing)} cached")

    # Load model
    t0 = time.time()
    model = _build_hyperiqa_model("/hyperiqa_vol/hyperiqa_plus_plus_best.pt")
    model = model.to("cuda").eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Preprocessing: 1600x1600, ImageNet normalization
    transform = T.Compose([
        T.Resize((1600, 1600)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = []
    for item in tqdm(dataset, desc="HyperIQA++"):
        img_id = item["image_id"]
        if img_id in existing:
            results.append(existing[img_id])
            continue

        try:
            pil_img = Image.open(item["image_path"]).convert("RGB")
            tensor = transform(pil_img).unsqueeze(0).to("cuda")

            with torch.no_grad():
                outputs = model(tensor)

            # Scores are already in [1,5] from soft label bin centers
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": {
                    "overall": outputs["overall"]["score"].item(),
                    "sharpness": outputs["sharpness"]["score"].item(),
                    "color": outputs["color"]["score"].item(),
                },
                "ground_truth": item["synthetic_scores"],
            }
        except Exception as exc:
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": None,
                "ground_truth": item["synthetic_scores"],
                "error": str(exc),
            }

        results.append(result)
        append_checkpoint(ckpt_path, result)

    # Sanity check
    print("\nSanity check (first 5 predictions):")
    for r in results[:5]:
        if r.get("predicted"):
            p = r["predicted"]
            g = r["ground_truth"]
            print(
                f"  {r['image_id']}: pred O={p['overall']:.2f} S={p['sharpness']:.2f} "
                f"C={p['color']:.2f} | gt O={g['overall']:.2f} S={g['sharpness']:.2f} "
                f"C={g['color']:.2f}"
            )

    # Flag MAE concern
    valid = [r for r in results if r.get("predicted")]
    if valid:
        import numpy as np

        preds = np.array([r["predicted"]["overall"] for r in valid])
        gts = np.array([r["ground_truth"]["overall"] for r in valid])
        mae = float(np.mean(np.abs(preds - gts)))
        mean_pred = float(np.mean(preds))
        print(f"\nScale check: mean_pred={mean_pred:.2f}, MAE={mae:.2f}")
        if mae > 1.5:
            print("WARNING: MAE > 1.5 suggests output scale mismatch!")

    results_volume.commit()
    return results


# ============================================================================
# DeQA-Doc Specialist Models
# ============================================================================


@app.function(
    image=deqa_image,
    gpu="L4",
    timeout=3600,
    volumes={
        "/synthetic": synthetic_volume,
        "/results": results_volume,
        "/deqa_checkpoints": deqa_volume,
    },
)
def evaluate_deqa_doc() -> list[dict]:
    """Evaluate DeQA-Doc 3 specialists on synthetic OOD dataset.

    Loads each specialist (overall, sharpness, color) sequentially,
    runs inference on all 520 images, then frees memory.

    CRITICAL: Level ordering is [excellent,good,fair,poor,bad] = [5,4,3,2,1].
    MOS = np.inner(probs, [5,4,3,2,1]).
    """
    import gc
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

    dataset = load_synthetic_dataset()
    ckpt_path = f"{CHECKPOINT_BASE}/deqa_doc_3specialists.jsonl"
    existing = load_checkpoint(ckpt_path)
    print(f"DeQA-Doc: {len(dataset)} images, {len(existing)} cached")

    # If all cached, return early
    if len(existing) == len(dataset):
        print("All images cached from checkpoint")
        return list(existing.values())

    # Level tokens and MOS weights
    level_names = ["excellent", "good", "fair", "poor", "bad"]
    mos_weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    # Model path — try ModelScope download or checkpoint volume
    # #ASSUME: model_availability: zhalala/DeQA-Doc on ModelScope contains
    #   the specialist models. If not, check /checkpoints/ volume.
    # #VERIFY: test modelscope download before full run
    model_path = _resolve_deqa_model_path()

    # Dimensions to evaluate, each with its own specialist
    dimensions = ["overall", "sharpness", "color_fidelity"]

    # Collect per-dimension predictions keyed by image_id
    dim_predictions: dict[str, dict[str, float]] = {}

    for dim in dimensions:
        print(f"\n{'=' * 50}")
        print(f"Loading specialist: {dim}")
        print(f"{'=' * 50}")

        # Determine specialist model path
        # Map dimension name to checkpoint directory name
        dim_dir_name = "color" if dim == "color_fidelity" else dim
        specialist_path = f"{model_path}/deqa_0618_{dim_dir_name}_norm_pair_1024"
        if not _path_exists(specialist_path):
            # Try simple {dim} subdirectory
            specialist_path = f"{model_path}/{dim}" if "/" in model_path else model_path
        if not _path_exists(specialist_path):
            # Fall back to single model for all dimensions
            specialist_path = model_path
            print(f"  Using shared model at {specialist_path}")

        t0 = time.time()

        # Disable torch default init for faster loading
        torch.nn.Linear.reset_parameters = lambda self: None  # type: ignore[assignment]
        torch.nn.LayerNorm.reset_parameters = lambda self: None  # type: ignore[assignment]

        model_name = get_model_name_from_path(specialist_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            specialist_path,
            None,  # model_base
            model_name,
            load_8bit=False,
            load_4bit=True,
            device="cuda",
            preprocessor_path="/root/deqa/preprocessor",
        )
        print(f"  Loaded in {time.time() - t0:.1f}s")

        # Get level token IDs
        level_ids = [tokenizer(tok)["input_ids"][1] for tok in level_names]
        print(f"  Level token IDs: {dict(zip(level_names, level_ids))}")

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

        dim_key = "color" if dim == "color_fidelity" else dim

        for item in tqdm(dataset, desc=f"DeQA [{dim}]"):
            img_id = item["image_id"]
            if img_id in existing:
                continue
            if img_id not in dim_predictions:
                dim_predictions[img_id] = {}
            if dim_key in dim_predictions[img_id]:
                continue

            try:
                pil_img = Image.open(item["image_path"]).convert("RGB")

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

                # Extract quality level probabilities
                level_logits = logits[0, level_ids]
                probs = torch.softmax(level_logits, dim=0).cpu().numpy()
                mos = float(np.inner(probs, mos_weights))

                dim_predictions[img_id][dim_key] = mos

            except Exception as exc:
                print(f"  ERROR {img_id}: {exc}")
                dim_predictions[img_id][dim_key] = float("nan")

        # Free GPU memory before loading next specialist
        del model, tokenizer, image_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("  GPU memory freed")

    # Combine predictions and write checkpoints
    results = []
    for item in dataset:
        img_id = item["image_id"]
        if img_id in existing:
            results.append(existing[img_id])
            continue

        preds = dim_predictions.get(img_id, {})
        if preds and not any(np.isnan(v) for v in preds.values()):
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": {
                    "overall": preds.get("overall"),
                    "sharpness": preds.get("sharpness"),
                    "color": preds.get("color"),
                },
                "ground_truth": item["synthetic_scores"],
            }
        else:
            result = {
                "image_id": img_id,
                "category": item["category"],
                "is_ood": item["is_ood"],
                "predicted": None,
                "ground_truth": item["synthetic_scores"],
                "error": "prediction failed",
            }

        results.append(result)
        append_checkpoint(ckpt_path, result)

    # Sanity check
    print("\nSanity check (first 5 predictions):")
    for r in results[:5]:
        if r.get("predicted"):
            p = r["predicted"]
            g = r["ground_truth"]
            print(
                f"  {r['image_id']}: pred O={p['overall']:.2f} S={p['sharpness']:.2f} "
                f"C={p['color']:.2f} | gt O={g['overall']:.2f} S={g['sharpness']:.2f} "
                f"C={g['color']:.2f}"
            )

    results_volume.commit()
    return results


def _resolve_deqa_model_path() -> str:
    """Find DeQA-Doc model path from checkpoint volume or ModelScope.

    Checks the deqa-specialist-checkpoints volume first (pre-uploaded),
    then falls back to ModelScope download.
    """
    import os

    # Check volume mount first
    volume_base = "/deqa_checkpoints"
    expected_dirs = [
        "deqa_0618_overall_norm_pair_1024",
        "deqa_0618_sharpness_norm_pair_1024",
        "deqa_0618_color_norm_pair_1024",
    ]
    if all(os.path.isdir(f"{volume_base}/{d}") for d in expected_dirs):
        print(f"Found DeQA specialists on volume at {volume_base}")
        return volume_base

    # Fall back to ModelScope download
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


def _path_exists(path: str) -> bool:
    """Check if a path exists (works inside Modal container)."""
    import os

    return os.path.exists(path)


# ============================================================================
# Metrics Computation
# ============================================================================


@app.function(
    image=modern_image,
    timeout=600,
    volumes={
        "/synthetic": synthetic_volume,
        "/results": results_volume,
    },
)
def compute_all_metrics(
    siglip2_results: list[dict],
    hyperiqa_results: list[dict],
    deqa_results: list[dict] | None = None,
) -> dict:
    """Compute comprehensive metrics matching VLM eval output format."""
    import json

    import numpy as np
    from scipy import stats

    def _srcc(pred, true):
        return float(stats.spearmanr(pred, true).statistic)

    def _plcc(pred, true):
        return float(stats.pearsonr(pred, true).statistic)

    def _bootstrap_ci(pred, true, metric_fn, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
        rng = np.random.RandomState(seed)
        n = len(pred)
        point = metric_fn(pred, true)
        vals = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            try:
                v = metric_fn(pred[idx], true[idx])
                if not np.isnan(v):
                    vals.append(v)
            except (ValueError, FloatingPointError):
                continue
        if len(vals) < 30:
            return point, float("nan"), float("nan")
        return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def compute_subset_metrics(results, subset_name="all"):
        valid = [r for r in results if r.get("predicted") is not None]
        if subset_name == "id":
            valid = [r for r in valid if not r["is_ood"]]
        elif subset_name == "ood":
            valid = [r for r in valid if r["is_ood"]]

        n = len(valid)
        if n < 10:
            return {"n": n, "subset": subset_name}

        metrics = {"n": n, "subset": subset_name}

        for dim in ["overall", "sharpness", "color"]:
            pred = np.array([r["predicted"][dim] for r in valid])
            true = np.array([r["ground_truth"][dim] for r in valid])

            srcc, srcc_lo, srcc_hi = _bootstrap_ci(pred, true, _srcc)
            mae = float(np.mean(np.abs(pred - true)))
            bias = float(np.mean(pred - true))

            metrics[f"{dim}_srcc"] = round(srcc, 4)
            metrics[f"{dim}_srcc_ci"] = f"[{srcc_lo:.4f}, {srcc_hi:.4f}]"
            metrics[f"{dim}_mae"] = round(mae, 4)
            metrics[f"{dim}_bias"] = round(bias, 4)

        if "overall_srcc" in metrics:
            metrics["wsrcc"] = round(
                W_OVERALL * metrics["overall_srcc"]
                + W_SHARPNESS * metrics.get("sharpness_srcc", 0)
                + W_COLOR * metrics.get("color_srcc", 0),
                4,
            )

        return metrics

    def compute_per_category(results):
        valid = [r for r in results if r.get("predicted") is not None]
        categories: dict[str, list[tuple[float, float]]] = {}
        for r in valid:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((r["predicted"]["overall"], r["ground_truth"]["overall"]))

        cat_metrics = {}
        for cat, pairs in sorted(categories.items()):
            n = len(pairs)
            pred = np.array([p for p, _ in pairs])
            true = np.array([t for _, t in pairs])
            mae = float(np.mean(np.abs(pred - true)))
            bias = float(np.mean(pred - true))
            srcc = float(stats.spearmanr(pred, true).statistic) if n >= 5 else None

            cat_metrics[cat] = {
                "n": n,
                "is_ood": "ood" in cat,
                "srcc_overall": round(srcc, 4) if srcc is not None else None,
                "mae_overall": round(mae, 4),
                "bias_overall": round(bias, 4),
            }

        return cat_metrics

    # Process each model
    model_results = {
        "siglip2_iqa_base_86m": siglip2_results,
        "hyperiqa_plus_plus": hyperiqa_results,
    }
    if deqa_results:
        model_results["deqa_doc_3specialists"] = deqa_results

    all_metrics = {}
    for model_name, results in model_results.items():
        m_all = compute_subset_metrics(results, "all")
        m_id = compute_subset_metrics(results, "id")
        m_ood = compute_subset_metrics(results, "ood")
        cat = compute_per_category(results)

        all_metrics[model_name] = {
            "all": m_all,
            "in_distribution": m_id,
            "out_of_distribution": m_ood,
            "per_category": cat,
        }

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")
        print(f"  {'Subset':<20s} {'wSRCC':>7s} {'SRCC_O':>7s} {'SRCC_S':>7s} "
              f"{'SRCC_C':>7s} {'MAE_O':>7s}")
        print("  " + "-" * 70)
        for label, m in [("All", m_all), ("In-Dist", m_id), ("OOD", m_ood)]:
            if m.get("overall_srcc") is not None:
                print(
                    f"  {label:<20s} {m.get('wsrcc', 0):>7.4f} "
                    f"{m.get('overall_srcc', 0):>7.4f} "
                    f"{m.get('sharpness_srcc', 0):>7.4f} "
                    f"{m.get('color_srcc', 0):>7.4f} "
                    f"{m.get('overall_mae', 0):>7.4f}"
                )

    # Save to results volume
    output_path = "/results/finetuned_synthetic_eval_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to volume: {output_path}")

    results_volume.commit()
    return all_metrics


# ============================================================================
# Local Entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    model: str = "all",
    metrics_only: bool = False,
):
    """Run synthetic OOD benchmark.

    Args:
        model: Which model to evaluate (siglip2, hyperiqa, deqa, all).
        metrics_only: Only compute metrics from existing checkpoints.
    """
    import json
    from pathlib import Path

    print("=" * 70)
    print("Synthetic OOD Benchmark: Fine-Tuned IQA Models")
    print("=" * 70)

    run_siglip2 = model in ("all", "siglip2")
    run_hyperiqa = model in ("all", "hyperiqa")
    run_deqa = model in ("all", "deqa")

    siglip2_results: list[dict] = []
    hyperiqa_results: list[dict] = []
    deqa_results: list[dict] | None = None

    if metrics_only:
        print("\nMetrics-only mode: loading from checkpoints")
        # TODO: load from volume checkpoints
        print("Not yet implemented — run full eval first")
        return

    # Run SigLIP2 and HyperIQA in parallel (same image, independent)
    futures = []
    if run_siglip2:
        print("\nLaunching SigLIP2 evaluation...")
        futures.append(("siglip2", evaluate_siglip2.spawn()))
    if run_hyperiqa:
        print("Launching HyperIQA++ evaluation...")
        futures.append(("hyperiqa", evaluate_hyperiqa.spawn()))

    # Run DeQA-Doc (different image, can run in parallel too)
    if run_deqa:
        print("Launching DeQA-Doc evaluation...")
        futures.append(("deqa", evaluate_deqa_doc.spawn()))

    # Collect results
    for name, future in futures:
        print(f"\nWaiting for {name}...")
        result = future.get()
        if name == "siglip2":
            siglip2_results = result
        elif name == "hyperiqa":
            hyperiqa_results = result
        elif name == "deqa":
            deqa_results = result
        ok = sum(1 for r in result if r.get("predicted"))
        print(f"  {name}: {ok}/{len(result)} successful predictions")

    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = compute_all_metrics.remote(
        siglip2_results, hyperiqa_results, deqa_results
    )

    # Save locally
    output_dir = Path("results/vlm_teacher_eval/full_eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "finetuned_synthetic_eval_metrics.json"
    output_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nResults saved to: {output_path}")

    # Print comparison with VLM baselines
    vlm_path = output_dir / "synthetic_eval_metrics.json"
    if vlm_path.exists():
        vlm_metrics = json.loads(vlm_path.read_text())
        print(f"\n{'=' * 70}")
        print("COMPARISON: Fine-Tuned vs VLM Teachers (synthetic OOD)")
        print(f"{'=' * 70}")
        print(f"{'Model':<35s} {'wSRCC':>7s} {'ID':>7s} {'OOD':>7s} {'MAE':>7s}")
        print("-" * 70)

        # VLM results
        for model_id, m in vlm_metrics.items():
            short = model_id.split("/")[-1] if "/" in model_id else model_id
            a = m.get("all", {})
            i = m.get("in_distribution", {})
            o = m.get("out_of_distribution", {})
            print(
                f"  [VLM] {short:<30s} {a.get('wsrcc', 0):>7.4f} "
                f"{i.get('wsrcc', 0):>7.4f} {o.get('wsrcc', 0):>7.4f} "
                f"{a.get('overall_mae', 0):>7.4f}"
            )

        # Fine-tuned results
        for model_id, m in all_metrics.items():
            a = m.get("all", {})
            i = m.get("in_distribution", {})
            o = m.get("out_of_distribution", {})
            print(
                f"  [FT]  {model_id:<30s} {a.get('wsrcc', 0):>7.4f} "
                f"{i.get('wsrcc', 0):>7.4f} {o.get('wsrcc', 0):>7.4f} "
                f"{a.get('overall_mae', 0):>7.4f}"
            )

    print("\nDone!")
