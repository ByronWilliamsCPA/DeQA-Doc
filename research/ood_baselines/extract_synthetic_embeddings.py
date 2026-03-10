"""Extract SigLIP2 embeddings for 520 synthetic OOD images.

Runs on Modal with GPU to extract 768-dim backbone features (before task heads)
for all synthetic images. Builds a combined evaluation NPZ with ground truth
ID/OOD labels for fair OOD baseline comparison.

Usage:
    # Extract embeddings on Modal (~5 min, L4 GPU)
    uv run modal run research/ood_baselines/extract_synthetic_embeddings.py

    # Then run evaluation locally:
    cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py \
        --ood-labels ../research/ood_baselines/eval_id_ood.npz
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("extract-synthetic-embeddings")

synthetic_volume = modal.Volume.from_name("synthetic-ood-data")
siglip2_volume = modal.Volume.from_name("siglip2-iqa-results")

siglip2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.51.0",
        "accelerate>=1.0.0",
        "numpy<2.0",
        "Pillow",
        "tqdm",
    )
)

# ============================================================================
# Shared constants
# ============================================================================

SYNTHETIC_DATA_DIR = "/synthetic/ood_poc_test"
MODEL_ID = "google/siglip2-base-patch16-naflex"
CHECKPOINT_PATH = "/siglip2_vol/siglip2/siglip2_iqa_best.pt"


# ============================================================================
# SigLIP2 model (inlined from modal/benchmark_synthetic_ood.py)
# ============================================================================


def _build_siglip2_model(model_id: str = MODEL_ID):
    """Build multi-task SigLIP2 model architecture.

    Inlined from modal/benchmark_synthetic_ood.py:_build_siglip2_model.
    We need the full model to load the checkpoint, but only use the backbone
    for embedding extraction.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoModel

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
        """Multi-task SigLIP2 with backbone + task heads."""

        def __init__(self, bb, hds, htypes):
            super().__init__()
            self.backbone = bb
            self.heads = hds
            self._head_types = htypes
            for hname, hcfg in head_configs.items():
                if hcfg["type"] == "reg":
                    self.register_buffer(f"temp_{hname}", torch.tensor(1.0))

        def extract_features(self, pixel_values, spatial_shapes=None):
            """Extract 768-dim backbone features BEFORE task heads."""
            out = self.backbone.get_image_features(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
            )
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.mean(dim=1)
            else:
                return out

    return _MultiTaskModel(backbone, heads, head_types)


def _load_synthetic_metadata(data_dir: str = SYNTHETIC_DATA_DIR) -> list[dict]:
    """Load synthetic dataset metadata from JSONL."""
    import json
    from pathlib import Path

    meta_path = Path(data_dir) / "metadata.jsonl"
    images = []
    with meta_path.open() as f:
        for line in f:
            d = json.loads(line)
            d["image_path"] = str(Path(data_dir) / d["image_id"])
            images.append(d)
    return images


# ============================================================================
# Modal function: extract embeddings on GPU
# ============================================================================


@app.function(
    image=siglip2_image,
    gpu="L4",
    timeout=1200,
    volumes={
        "/synthetic": synthetic_volume,
        "/siglip2_vol": siglip2_volume,
    },
)
def extract_embeddings() -> dict:
    """Extract 768-dim SigLIP2 backbone features for all 520 synthetic images.

    Returns:
        Dict with keys: embeddings, image_names, categories, is_ood.
        Values are lists (serializable over Modal boundary).
    """
    import time

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor

    dataset = _load_synthetic_metadata()
    print(f"Loaded {len(dataset)} synthetic images")

    # Load model + checkpoint
    t0 = time.time()
    model = _build_siglip2_model(MODEL_ID)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cuda", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model = model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Extract embeddings
    all_embeddings = []
    all_names = []
    all_categories = []
    all_is_ood = []

    for item in tqdm(dataset, desc="Extracting embeddings"):
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
            features = model.extract_features(pixel_values, spatial_shapes)

        all_embeddings.append(features.cpu().numpy().squeeze(0))
        all_names.append(item["image_id"])
        all_categories.append(item["category"])
        all_is_ood.append(item["is_ood"])

    embeddings = np.stack(all_embeddings, axis=0)
    print(f"\nExtracted embeddings: {embeddings.shape}")
    print(f"  ID images: {sum(not x for x in all_is_ood)}")
    print(f"  OOD images: {sum(all_is_ood)}")

    return {
        "embeddings": embeddings.tolist(),
        "image_names": all_names,
        "categories": all_categories,
        "is_ood": all_is_ood,
    }


# ============================================================================
# Local entrypoint: save NPZ files
# ============================================================================


@app.local_entrypoint()
def main():
    """Extract embeddings on Modal, then build evaluation NPZ locally."""
    from pathlib import Path

    import numpy as np

    print("=" * 60)
    print("Extracting SigLIP2 embeddings for synthetic OOD images")
    print("=" * 60)

    # Run extraction on GPU
    result = extract_embeddings.remote()

    synth_emb = np.array(result["embeddings"], dtype=np.float32)
    synth_names = np.array(result["image_names"])
    synth_categories = np.array(result["categories"])
    synth_is_ood = np.array(result["is_ood"])

    print(f"\nReceived embeddings: {synth_emb.shape}")
    print(f"  ID: {(~synth_is_ood).sum()}, OOD: {synth_is_ood.sum()}")

    # Save synthetic embeddings
    output_dir = Path("research/ood_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)

    synth_path = output_dir / "synthetic_embeddings.npz"
    np.savez_compressed(
        synth_path,
        embeddings=synth_emb,
        image_names=synth_names,
        categories=synth_categories,
        is_ood=synth_is_ood,
    )
    print(f"Saved synthetic embeddings to {synth_path}")

    # Build combined evaluation NPZ
    test_path = Path("results/siglip2_diqa5000/embeddings/test.npz")
    if not test_path.exists():
        print(f"WARNING: {test_path} not found, skipping eval NPZ build")
        return

    test_data = np.load(test_path)
    test_emb = test_data["embeddings"]  # (1000, 768)
    print(f"Loaded test embeddings: {test_emb.shape}")

    # Combine: test (all ID) + synthetic (mixed ID/OOD)
    eval_emb = np.concatenate([test_emb, synth_emb], axis=0)
    eval_labels = np.concatenate([
        np.zeros(len(test_emb), dtype=int),  # All test = ID
        synth_is_ood.astype(int),  # Use ground truth is_ood
    ])
    eval_categories = np.concatenate([
        np.full(len(test_emb), "diqa_test"),
        synth_categories,
    ])

    n_id = int((eval_labels == 0).sum())
    n_ood = int((eval_labels == 1).sum())
    print(f"\nCombined eval set: {eval_emb.shape}")
    print(f"  ID: {n_id}, OOD: {n_ood}, Total: {len(eval_labels)}")

    eval_path = output_dir / "eval_id_ood.npz"
    np.savez_compressed(
        eval_path,
        embeddings=eval_emb,
        labels=eval_labels,
        categories=eval_categories,
    )
    print(f"Saved evaluation NPZ to {eval_path}")
    print("\nDone! Next steps:")
    print(f"  cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py \\")
    print(f"      --ood-labels ../research/ood_baselines/eval_id_ood.npz")
