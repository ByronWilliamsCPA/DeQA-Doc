# Handoff: Fine-Tuned IQA Model Evaluation on Synthetic OOD Dataset

**Date:** 2026-03-07
**From:** DeQA-Doc VLM Teacher Evaluation
**To:** image_detection team
**Priority:** P1 — blocks paper completion

---

## Objective

Evaluate 3 fine-tuned IQA models on the 520-image synthetic OOD dataset to measure
how well domain-trained models handle out-of-distribution documents. This completes
a direct comparison: we already have VLM teacher results on this dataset (3 models),
and need the fine-tuned model results to quantify the gap.

### Why This Matters

Our VLM teacher evaluation shows Gemini 3 Flash achieves wSRCC=0.738 on the synthetic
dataset. We need to know if the fine-tuned models (which score 0.716-0.886 on DIQA-5000)
maintain that advantage on OOD data, or if VLM teachers are actually better at
generalizing to unseen document types.

---

## The 3 Models to Evaluate

| # | Model | Architecture | DIQA-5000 wSRCC | Params | Expected GPU |
|---|-------|-------------|-----------------|--------|-------------|
| 1 | **SigLIP2-IQA-Base-86M-v1.0.0** | ViT-B/16 + 3 regression heads | 0.886 | 88M | T4/L4 (3GB) |
| 2 | **HyperIQA-Plus-Plus-DIQA5000** | ResNet-50 + HyperNet + SpatialAttn | 0.856 | 138M | T4/L4 (2GB) |
| 3 | **DeQA-Doc-3Specialists** | mPLUG-Owl2 (LLaMA-2 7B) x3 | 0.716 | 3x7B | A10/L4 (12GB) |

---

## Model Loading Details

### Model 1: SigLIP2-IQA-Base-86M

**Source code:**
- Production wrapper: `image_detection/src/image_preprocessing_detector/detection/siglip2_multitask.py`
- Training script: `image_detection/modal/train_siglip2_iqa_v2.py`
- Class: `SigLIP2MultiTaskDetector`

**Checkpoint:**
- Filename: `siglip2_iqa_best.pt`
- Location: Modal volume `dociq-checkpoints` or local training output
- Format: PyTorch state dict with keys `model_state_dict`, backbone + 8 task heads
- Note: Current checkpoint has architecture mismatch (445 missing keys, 368 unexpected)
  due to multi-task vs IQA-only head differences. Use `strict=False` on load.

**Loading pattern:**
```python
from image_preprocessing_detector.detection.siglip2_multitask import (
    SigLIP2MultiTaskDetector,
)

detector = SigLIP2MultiTaskDetector(
    checkpoint_path="path/to/siglip2_iqa_best.pt",
)
# Lazy-loads on first .predict() call
result = detector.predict(image_bgr)  # numpy BGR uint8

# Extract IQA scores
overall = result.iqa_overall.mu      # float, 0-1 range (needs rescaling to 1-5)
sharpness = result.iqa_sharpness.mu
color = result.iqa_color.mu
embedding = result.embedding          # 768-dim, for OOD detection
```

**Preprocessing (handled internally):**
- BGR/grayscale numpy -> RGB PIL
- AutoProcessor with `max_num_patches=784`, `padding="max_length"`
- Returns `pixel_values` and `spatial_shapes` tensors

**Output range:** The model outputs scores in [0, 1] range. You MUST rescale to [1, 5]
for comparison with human MOS: `mos = mu * 4.0 + 1.0` (verify this mapping against
the training script's label normalization).

**Dependencies:**
```
torch>=2.0
transformers>=4.36
```

### Model 2: HyperIQA-Plus-Plus-DIQA5000

**Source code:**
- Training script: `image_detection/modal/train_hyperiqa_plus_plus.py` (check git history — may be removed from current branch, `.pyc` cache exists)
- Model card: `image_detection/docs/model-cards/production/hyperiqa_plus_plus_diqa5000.md`

**Checkpoint:**
- Filename: `hyperiqa_plus_plus_best.pt`
- Location: Modal volume `dociq-checkpoints` or `image_detection/models/hyperiqa_plus_plus/`
- Format: Full PyTorch model (not just state dict — verify)

**Architecture:**
- ResNet-50 backbone (from `pyiqa` HyperIQA)
- HyperNet for content-adaptive feature processing
- Spatial attention (DocIQ-simplified) for layout-aware weighting
- 3 soft-label distribution heads (10-bin each) for overall, sharpness, color
- Input size: **1600x1600x3** (critical — this model expects high-resolution input)

**Loading pattern:**
```python
import pyiqa
import torch

# Option A: If using pyiqa's HyperIQA as base
model = pyiqa.create_metric('hyperiqa')
# Then load custom checkpoint on top
checkpoint = torch.load("hyperiqa_plus_plus_best.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"], strict=False)

# Option B: If standalone model class exists in training script
# Check train_hyperiqa_plus_plus.py for the model class definition
```

**Preprocessing:**
- Resize to 1600x1600 (preserving aspect ratio with padding, or center crop — check training script)
- Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

**Output format:** 10-bin soft label distribution per dimension.
- Convert to MOS: `score = sum(prob_i * center_i for i in range(10))`
- Bin centers depend on training config (check training script for bin edge definitions)

**Output range:** Likely MOS [1, 5] directly from soft label expectation. Verify.

**Dependencies:**
```
torch>=2.0
pyiqa>=0.1.10
timm
```

**Known issue:** MAE of 2.225 on DIQA-5000 is very high despite good SRCC/PLCC.
This suggests a systematic scale offset — the model's output range may not be
properly calibrated to [1, 5]. Check and apply appropriate rescaling.

### Model 3: DeQA-Doc-3Specialists

**Source code:**
- Inference script: `DeQA-Doc/DeQA-Score/src/evaluate/iqa_eval.py`
- Model builder: `DeQA-Doc/DeQA-Score/src/model/builder.py`
- Conversation template: `DeQA-Doc/DeQA-Score/src/conversation.py`

**Checkpoint:**
- 3 separate models, one per quality dimension (overall, sharpness, color_fidelity)
- Base: `MAGAer13/mplug-owl2-llama2-7b` from HuggingFace
- Fine-tuned weights: Check `DeQA-Score/scripts/infer.sh` for model paths
- Each model is ~14GB (7B params in FP16)

**Loading pattern:**
```python
from src.model.builder import load_pretrained_model
from src.conversation import conv_templates
from src.constants import DEFAULT_IMAGE_TOKEN
from src.mm_utils import tokenizer_image_token, expand2square

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="path/to/specialist_overall",
    model_base=None,
    model_name="mplug-owl2",
    load_8bit=False,
    load_4bit=True,  # Recommended to fit on A10
    device="cuda",
)

# Inference
image = Image.open(path).convert("RGB")
image = expand2square(image, (255, 255, 255))
image_tensor = image_processor.preprocess(image)["pixel_values"][0].unsqueeze(0).half().to("cuda")

conv = conv_templates["mplug_owl2"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nHow would you rate the quality of this image?")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt() + " The quality of the image is"

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

with torch.inference_mode():
    output = model(input_ids=input_ids, images=image_tensor)
    logits = output["logits"][:, -1, :]

# Extract quality level probabilities
level_tokens = ["excellent", "good", "fair", "poor", "bad"]
level_ids = [tokenizer(tok).input_ids[-1] for tok in level_tokens]
probs = torch.softmax(logits[0, level_ids], dim=0).cpu().numpy()

# CRITICAL: DeQA convention is [excellent, good, fair, poor, bad] = [5, 4, 3, 2, 1]
mos = np.inner(probs, [5, 4, 3, 2, 1])
```

**Important notes:**
- This model predicts ONE dimension per specialist. You need 3 separate model loads.
- Each load takes ~30-60s and uses ~12GB VRAM (4-bit quantized).
- Running all 3 sequentially: ~15 minutes for 520 images.
- The level ordering is `[excellent, good, fair, poor, bad]` = `[5, 4, 3, 2, 1]`.
  Getting this wrong inverts the correlation. Verify against `DeQA-Score/src/train/loss.py:25`.

**Dependencies:**
```
torch==2.0.1
transformers==4.36.1
peft==0.4.0
accelerate==0.21.0
```

---

## Synthetic OOD Dataset

### Location

The dataset is generated by `DeQA-Doc/DeQA-Score/src/uncertainty/generate_ood_poc_dataset.py`.
For the eval run, images should be at a known path (we used `/tmp/ood_poc_test/`).

**To regenerate or download:** The generation script creates images from document
templates with controlled degradation parameters. If images aren't available on Modal,
you'll need to either:
1. Upload the 520 images to a Modal volume (~200MB), or
2. Run the generation script on Modal first (requires document templates)

### Structure

```
/tmp/ood_poc_test/
  metadata.jsonl          # Ground truth (one JSON per line)
  manifest.json           # Dataset summary
  id_standard/            # 100 images — in-distribution (Latin)
  id_cyrillic/            # 50 images — in-distribution (Cyrillic)
  ood_script_tibetan/     # 30 images
  ood_script_myanmar/     # 30 images
  ood_script_ethiopic/    # 30 images
  ood_adversarial_fraktur/    # 20 images
  ood_adversarial_nastaliq/   # 20 images
  ood_binarized/          # 30 images
  ood_cjk_vertical/       # 30 images
  ood_form_layout/        # 30 images
  ood_heavily_degraded/   # 30 images
  ood_multiscript/        # 30 images
  ood_pristine/           # 30 images
  ood_very_high_dpi/      # 30 images
  ood_very_low_dpi/       # 30 images
```

### Ground Truth Format

Each line of `metadata.jsonl`:
```json
{
  "image_id": "id_standard/id_standard_0000.jpg",
  "image_path": "/tmp/ood_poc_test/id_standard/id_standard_0000.jpg",
  "category": "id_standard",
  "is_ood": false,
  "synthetic_scores": {
    "overall": 2.83,
    "sharpness": 3.19,
    "color": 2.66
  }
}
```

**Score range:** 1.0-5.0 (MOS scale), derived from generation parameters.

**Score field mapping:**
- `synthetic_scores.overall` -> overall quality
- `synthetic_scores.sharpness` -> sharpness
- `synthetic_scores.color` -> color fidelity

---

## Required Metrics

Compute these metrics to match our VLM evaluation format exactly:

### Per-Dimension (overall, sharpness, color_fidelity)

| Metric | Formula | Notes |
|--------|---------|-------|
| SRCC | `scipy.stats.spearmanr(pred, gt).statistic` | Primary ranking metric |
| SRCC 95% CI | Bootstrap 1000 iterations, seed=42 | `np.percentile(boot_srcc, [2.5, 97.5])` |
| PLCC | `scipy.stats.pearsonr(pred, gt).statistic` | Linear correlation |
| MAE | `np.mean(np.abs(pred - gt))` | Absolute error |
| Bias | `np.mean(pred - gt)` | Systematic over/under-rating |

### Aggregate

| Metric | Formula |
|--------|---------|
| wSRCC | `0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color` |

### Subsets

Compute all metrics for 3 subsets:
1. **All** (n=520)
2. **In-distribution** (n=150): categories where `is_ood=false`
3. **Out-of-distribution** (n=370): categories where `is_ood=true`

### Per-Category

For each of the 15 categories, compute overall SRCC and MAE (minimum n=20 per category).

---

## Existing VLM Results for Comparison

These are the results to compare against (from `results/vlm_teacher_eval/full_eval/results/synthetic_eval_metrics.json`):

| Model | wSRCC (all) | wSRCC (ID) | wSRCC (OOD) | SRCC_O (all) |
|-------|-------------|------------|-------------|--------------|
| GPT-4.1 | 0.757 | 0.751 | 0.747 | 0.764 |
| Gemini 3 Flash | 0.738 | 0.752 | 0.745 | 0.753 |
| Claude Haiku 4.5 | 0.591 | 0.526 | 0.646 | 0.582 |

**Question we're answering:** Do fine-tuned models (0.716-0.886 on DIQA-5000) maintain
their advantage on OOD data, or do VLM teachers generalize better to unseen document types?

---

## Proposed Modal App Structure

### Option A: Extend Existing Infrastructure

The `image_detection/modal/` directory already has the Modal app framework:
- `modal/app.py` — shared Modal app definition
- `modal/shared/constants.py` — volumes, GCS config, secrets
- `modal/shared/gcs_utils.py` — dataset download utilities
- `modal/shared/metrics_utils.py` — metric computation

**Recommended approach:** Create `modal/benchmark_synthetic_ood.py` following the
existing patterns, reusing shared volumes and utilities.

### Option B: Standalone Script

```python
import modal

app = modal.App("synthetic-ood-benchmark")

# Reuse existing volumes for model checkpoints
checkpoint_volume = modal.Volume.from_name("dociq-checkpoints")

# Create new volume for synthetic dataset
synthetic_volume = modal.Volume.from_name("synthetic-ood-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision",
        "transformers>=4.36",
        "scipy",
        "pillow",
        "numpy",
        "pyiqa",
        "timm",
    )
)

@app.function(
    gpu="l4",           # 24GB — fits all 3 models
    timeout=3600,       # 1 hour
    volumes={
        "/checkpoints": checkpoint_volume,
        "/synthetic": synthetic_volume,
    },
    image=image,
)
def evaluate_model(model_name: str) -> dict:
    """Load one model, run inference on all 520 images, return predictions."""
    ...

@app.function(timeout=300)
def compute_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compute SRCC/PLCC/MAE per dimension with bootstrap CIs."""
    ...

@app.local_entrypoint()
def main():
    for model in ["siglip2", "hyperiqa", "deqa_overall", "deqa_sharpness", "deqa_color"]:
        preds = evaluate_model.remote(model)
        metrics = compute_metrics.remote(preds, ground_truth)
        print(f"{model}: {metrics}")
```

### GPU Recommendation

| Configuration | GPU | Est. Time | Est. Cost |
|--------------|-----|-----------|-----------|
| All 3 models serial | L4 (24GB) | ~15 min | $0.15 |
| SigLIP2 + HyperIQA only | T4 (16GB) | ~3 min | $0.02 |
| DeQA-Doc alone | A10 (24GB) | ~12 min | $0.20 |

**Recommended:** L4 GPU, run SigLIP2 and HyperIQA first (fast, low memory), then
load DeQA-Doc specialists one at a time.

---

## Output Format

Save results as JSON matching our existing format:

```json
{
  "siglip2_iqa_base_86m": {
    "all": {
      "n": 520,
      "subset": "all",
      "overall_srcc": 0.XXX,
      "overall_srcc_ci": "[X.XXXX, X.XXXX]",
      "overall_mae": 0.XXX,
      "overall_bias": 0.XXX,
      "sharpness_srcc": 0.XXX,
      "sharpness_srcc_ci": "[X.XXXX, X.XXXX]",
      "sharpness_mae": 0.XXX,
      "sharpness_bias": 0.XXX,
      "color_srcc": 0.XXX,
      "color_srcc_ci": "[X.XXXX, X.XXXX]",
      "color_mae": 0.XXX,
      "color_bias": 0.XXX,
      "wsrcc": 0.XXX
    },
    "in_distribution": { ... },
    "out_of_distribution": { ... },
    "per_category": {
      "id_standard": { "n": 100, "is_ood": false, "srcc_overall": 0.XXX, "mae_overall": 0.XXX },
      ...
    }
  },
  "hyperiqa_plus_plus": { ... },
  "deqa_doc_3specialists": { ... }
}
```

Save to: `DeQA-Doc/results/vlm_teacher_eval/full_eval/results/finetuned_synthetic_eval_metrics.json`

Also save per-sample predictions as JSONL checkpoints:
- `checkpoints_synthetic/siglip2_iqa_base_86m.jsonl`
- `checkpoints_synthetic/hyperiqa_plus_plus.jsonl`
- `checkpoints_synthetic/deqa_doc_3specialists.jsonl`

---

## Known Blockers and Verification Steps

### Before running, verify:

1. **SigLIP2 checkpoint available on Modal volume**
   ```bash
   # Check if checkpoint exists
   modal volume get dociq-checkpoints siglip2_iqa_best.pt
   ```

2. **HyperIQA checkpoint available**
   ```bash
   # Check training script in git history if removed from current branch
   git log --all --oneline -- modal/train_hyperiqa_plus_plus.py
   git show <commit>:modal/train_hyperiqa_plus_plus.py
   ```

3. **DeQA-Doc specialist model paths**
   ```bash
   # Check infer.sh for the exact model paths used in competition
   cat DeQA-Doc/DeQA-Score/scripts/infer.sh
   ```

4. **SigLIP2 output range** — Confirm whether mu output is [0,1] or [1,5]:
   ```python
   # Quick test: run on one DIQA-5000 image with known MOS
   # If output ~0.6 for a MOS=3.0 image, it's [0,1] range
   # If output ~3.0, it's already [1,5]
   ```

5. **HyperIQA MAE concern** — The DIQA-5000 benchmark shows MAE=2.225 which is
   suspiciously high for a model with PLCC=0.886. This may indicate a scale mismatch.
   Run a sanity check on 5 DIQA-5000 images before the full synthetic run.

6. **Synthetic dataset upload** — Upload the 520 images to a Modal volume:
   ```bash
   # From the machine with /tmp/ood_poc_test/
   tar czf synthetic_ood.tar.gz -C /tmp ood_poc_test/
   modal volume put synthetic-ood-data synthetic_ood.tar.gz
   ```

### DeQA-Doc level ordering — CRITICAL

The DeQA convention maps quality levels as:
```
[excellent, good, fair, poor, bad] = [5, 4, 3, 2, 1]
```

The MOS reconstruction is: `np.inner(probs, [5, 4, 3, 2, 1])`

Getting this backwards inverts the correlation (SRCC becomes negative). This is
confirmed in `DeQA-Score/src/train/loss.py:25` and `src/evaluate/cal_distribution_gap.py:79`.

---

## Timeline

| Step | Est. Time | Who |
|------|-----------|-----|
| Upload synthetic dataset to Modal volume | 5 min | Executor |
| Verify all 3 checkpoints available | 10 min | Executor |
| Write Modal benchmark script | 1-2 hours | Executor |
| Run SigLIP2 + HyperIQA (fast) | 5 min | Modal |
| Run DeQA-Doc 3 specialists (slow) | 12 min | Modal |
| Compute metrics and verify | 15 min | Executor |
| Return results JSON | — | Executor |

**Total estimated effort:** ~2-3 hours (mostly script writing)
**Total compute cost:** ~$0.15-0.35

---

## Contact

Results should be saved as specified above and committed to the `DeQA-Doc` repository
under `results/vlm_teacher_eval/full_eval/`. The VLM teacher evaluation paper
(`VLM_TEACHER_EVALUATION.md`) will be updated to include the comparison.

Questions about metric computation: reference `run_synthetic_eval.py` in the same directory.
Questions about the synthetic dataset: reference `generate_ood_poc_dataset.py` in `DeQA-Score/src/uncertainty/`.
