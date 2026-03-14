# Handoff: OOD Baseline Re-Evaluation with Ground Truth Labels

**Priority**: High | **Effort**: Low | **Est. compute cost**: ~$0.50 (Modal L4, ~5 min embedding extraction)
**Addresses**: Paper 4 peer review (unanimous) + RESULTS.md limitation #1-2
**Depends on**: None (all inputs exist or can be generated)
**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

---

## Objective

Re-run the OOD baseline comparison (Mahalanobis, k-NN, cosine, energy) with **proper ground truth OOD labels** instead of the circular proxy labels currently used. The existing evaluation used Mahalanobis-derived pseudo-labels to evaluate Mahalanobis itself, making its reported 0.9999 AUROC meaningless. This handoff fixes that by: (1) extracting SigLIP2 embeddings for the 520 synthetic OOD images, and (2) re-running the existing evaluation scripts against real ID vs OOD labels.

## Why This Matters

The current OOD baseline results in `research/ood_baselines/RESULTS.md` have three documented limitations:

1. **Circular evaluation**: Mahalanobis AUROC (0.9999) is computed against labels derived from the Mahalanobis detector itself — this is not a valid evaluation
2. **No real OOD data**: Synthetic OOD embeddings were never extracted, so the comparison used only DIQA-5000 test images with proxy labels
3. **Paper 4 blocker**: Peer review unanimously flagged "no baseline OOD method comparisons on same embeddings" — the current results only partially address this due to the circular labeling

With proper ground truth (DIQA-5000 test = ID, synthetic 520 = OOD), we get a fair comparison of all four methods on genuinely different distributions.

## What Already Exists

### Completed code (ready to re-run)

| File | Purpose | Status |
|------|---------|--------|
| [ood_methods.py](../ood_baselines/ood_methods.py) | 4 scoring functions (Mahalanobis, k-NN, cosine, energy) | Done, no changes needed |
| [evaluate_ood.py](../ood_baselines/evaluate_ood.py) | Evaluation with AUROC/AUPRC/FPR metrics + figures | Done, has `--ood-labels` flag ready |
| [knn_sensitivity.py](../ood_baselines/knn_sensitivity.py) | k-NN sweep over k={1,3,5,10,20,50,100} | Done, needs real labels support |

### Completed artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Train embeddings (3,500 x 768) | `results/siglip2_diqa5000/embeddings/train.npz` | Done |
| Val embeddings (500 x 768) | `results/siglip2_diqa5000/embeddings/val.npz` | Done |
| Test embeddings (1,000 x 768) | `results/siglip2_diqa5000/embeddings/test.npz` | Done |
| Pre-fitted Mahalanobis detector | `results/siglip2_diqa5000/ood_detector_v2.npz` | Done |
| Proxy-label results | `research/ood_baselines/ood_baseline_results.json` | Done (to be replaced) |
| Proxy-label figures | `research/ood_baselines/figures/` | Done (to be replaced) |

### Missing (this handoff creates)

| Artifact | Description |
|----------|-------------|
| Synthetic OOD embeddings | SigLIP2 768-dim embeddings for all 520 synthetic images |
| Evaluation NPZ | Combined ID + OOD embeddings with binary ground truth labels |
| Updated results | Fair AUROC/AUPRC/FPR comparison without circular evaluation |

## Approach

### Step 1: Generate Synthetic OOD Images (~1 min)

The 520 synthetic images are generated on-demand. Run the generation:

```bash
cd DeQA-Score
.venv/bin/python ../results/vlm_teacher_eval/full_eval/run_synthetic_eval.py --generate-only
```

Or use the Modal benchmark script which also generates them:
```bash
uv run modal run modal/benchmark_synthetic_ood.py --model siglip2
```

Images land in `/tmp/ood_poc_test/` with `metadata.jsonl` containing per-image category labels and `is_ood` flags.

**Important**: If neither script has a `--generate-only` mode, check how `benchmark_synthetic_ood.py` calls `load_synthetic_dataset()` — the generation logic is embedded there. You may need to extract just the generation portion.

### Step 2: Extract SigLIP2 Embeddings for Synthetic OOD (~5 min, GPU)

Create a script `research/ood_baselines/extract_synthetic_embeddings.py` that:

1. Loads the SigLIP2-IQA-Base model (same as `benchmark_synthetic_ood.py` lines 181-261)
2. Processes all 520 synthetic images through the backbone
3. Extracts 768-dim embeddings (mean pooling over sequence dim, matching how DIQA-5000 embeddings were extracted)
4. Saves to NPZ format

**Model loading reference** (from `benchmark_synthetic_ood.py`):
```python
from transformers import AutoModel, AutoProcessor

model_id = "google/siglip2-base-patch16-naflex"
model = _build_siglip2_model(model_id)  # See lines 181-261
checkpoint = torch.load("/path/to/siglip2_iqa_best.pt", map_location="cuda", weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)
model = model.to("cuda").eval()
processor = AutoProcessor.from_pretrained(model_id)
```

**Embedding extraction** (after forward pass, before heads):
```python
inputs = processor(images=pil_img, return_tensors="pt", max_num_patches=784, padding="max_length")
pixel_values = inputs["pixel_values"].to("cuda")
spatial_shapes = inputs["spatial_shapes"].to("cuda")

with torch.no_grad():
    out = model.backbone.get_image_features(pixel_values=pixel_values, spatial_shapes=spatial_shapes)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        embedding = out.pooler_output  # (1, 768)
    else:
        embedding = out.last_hidden_state.mean(dim=1)  # (1, 768)
```

**Output format**:
```
research/ood_baselines/synthetic_embeddings.npz
  - embeddings: (520, 768) float32
  - image_names: (520,) unicode
  - categories: (520,) unicode  (e.g., "ood_script_tibetan")
  - is_ood: (520,) bool
```

**Checkpoint location**: Modal volume `dociq-checkpoints` at path `siglip2/siglip2_iqa_best.pt`. If running locally, download first. If running on Modal, mount the volume (see `benchmark_synthetic_ood.py` for volume mounts).

### Step 3: Build Evaluation NPZ

Combine DIQA-5000 test (ID) + synthetic OOD into a single evaluation set:

```python
# Load ID
test = np.load("results/siglip2_diqa5000/embeddings/test.npz")
id_emb = test["embeddings"]       # (1000, 768)
id_names = test["image_names"]    # (1000,)
id_labels = np.zeros(1000, dtype=int)  # All ID = 0

# Load OOD
synth = np.load("research/ood_baselines/synthetic_embeddings.npz")
ood_emb = synth["embeddings"]     # (520, 768)
ood_names = synth["image_names"]  # (520,)
ood_labels = np.ones(520, dtype=int)  # All OOD = 1

# Note: 150 of the 520 synthetic images are actually ID (id_standard, id_cyrillic).
# Use the is_ood field from metadata for correct labels:
ood_labels = synth["is_ood"].astype(int)  # 370 OOD + 150 ID

# Combine
eval_emb = np.concatenate([id_emb, ood_emb])          # (1520, 768)
eval_labels = np.concatenate([id_labels, ood_labels])  # (1520,)

# Save
np.savez_compressed(
    "research/ood_baselines/eval_id_ood.npz",
    embeddings=eval_emb,
    labels=eval_labels,
    categories=np.concatenate([np.full(1000, "diqa_test"), synth["categories"]]),
)
```

**Important nuance**: The synthetic dataset has 150 ID images (100 `id_standard` + 50 `id_cyrillic`) and 370 OOD images. Use `metadata.jsonl`'s `is_ood` field — do NOT label all 520 as OOD. The final evaluation set should be:
- ID: 1,000 (DIQA test) + 150 (synthetic ID) = 1,150
- OOD: 370 (synthetic OOD)
- Total: 1,520

### Step 4: Re-run Evaluation

The existing `evaluate_ood.py` already supports `--ood-labels`:

```bash
cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py \
    --ood-labels ../research/ood_baselines/eval_id_ood.npz
```

**However**, the `load_real_ood_labels()` function (line 119) currently expects only `embeddings` and `labels` keys. It will need a minor update to also accept `categories` for per-category breakdown. See Deliverable 2 below.

### Step 5: Re-run k-NN Sensitivity

Update `knn_sensitivity.py` to accept `--ood-labels` (same pattern as evaluate_ood.py) and re-sweep:

```bash
cd DeQA-Score && .venv/bin/python ../research/ood_baselines/knn_sensitivity.py \
    --ood-labels ../research/ood_baselines/eval_id_ood.npz
```

## Deliverables

### 1. Embedding extraction: `research/ood_baselines/extract_synthetic_embeddings.py`

Script that loads SigLIP2, processes 520 synthetic images, and saves embeddings + metadata to NPZ. Can run on Modal (preferred) or locally with GPU.

### 2. Updated evaluate_ood.py

Minor changes:
- `load_real_ood_labels()` should also return `categories` array
- Add per-category AUROC breakdown (the handoff 02 spec already requested this)
- Update `write_results_md()` to include per-category results table
- Update `write_results_json()` to include `per_category` section

### 3. Updated knn_sensitivity.py

Add `--ood-labels` argument matching `evaluate_ood.py`'s interface.

### 4. Per-category heatmap: `research/ood_baselines/figures/per_category_heatmap.png`

Method (rows) x OOD category (columns) AUROC heatmap. This was in the original handoff 02 spec but couldn't be computed with proxy labels.

### 5. Updated results: `research/ood_baselines/ood_baseline_results.json`

Replace proxy results with ground truth results. Schema additions:

```json
{
  "label_source": "ground_truth",
  "eval_set": {"n_id": 1150, "n_ood": 370, "total": 1520},
  "methods": {
    "mahalanobis": {"auroc": ..., "auroc_ci": [...], ...},
    "knn_k10": {...},
    "cosine": {...},
    "energy": {...}
  },
  "knn_sensitivity": {"k1": ..., "k3": ..., ...},
  "per_category": {
    "ood_script_tibetan": {"mahalanobis": ..., "knn_k10": ..., ...},
    "ood_heavily_degraded": {...},
    ...
  }
}
```

### 6. Updated RESULTS.md

Replace the proxy-label results and remove the "circular evaluation" caveats. Include:
- Main comparison table (4 methods)
- Per-category breakdown table
- k-NN sensitivity (with real labels)
- Updated recommendation (does Mahalanobis still dominate with fair evaluation?)

### 7. Updated figures

Re-generate all existing figures + add per-category heatmap:
- `figures/roc_comparison.png` — with ground truth labels
- `figures/auroc_bar_comparison.png` — with ground truth labels
- `figures/knn_k_sensitivity.png` — with ground truth labels
- `figures/per_category_heatmap.png` — NEW

## Key Questions for the Team

1. **Embedding consistency**: Verify that the embedding extraction process matches how `train.npz`/`val.npz`/`test.npz` were created. The pooling strategy (pooler_output vs mean pooling) must be identical. Check the original extraction code in `image_detection/modal/train_siglip2_iqa_v2.py` if available.

2. **Synthetic image persistence**: The synthetic images are generated in `/tmp/` and are ephemeral. If running on Modal, ensure the generation and extraction happen in the same function call, or persist images to a Modal volume first.

3. **Threshold recalibration**: After getting real AUROC numbers, the production threshold (currently 46.0) may need adjustment. Report the optimal threshold at 95% TPR and 99% TPR for the best-performing method.

## Dependencies

### For embedding extraction (Step 2)
```
torch>=2.5.1, transformers>=4.51.0, Pillow, tqdm, numpy<2.0
```
Requires GPU (Modal L4 recommended, ~$0.50 for 5 min).

### For evaluation (Steps 3-5)
```
numpy, scipy, scikit-learn, matplotlib, seaborn (for heatmap)
```
All available in `DeQA-Score/.venv/`. No GPU needed.

## Definition of Done

- [ ] SigLIP2 embeddings extracted for all 520 synthetic images
- [ ] Evaluation NPZ created with correct ID/OOD labels (1,150 ID + 370 OOD)
- [ ] All 4 methods evaluated with ground truth labels (AUROC, AUPRC, FPR@95/99TPR)
- [ ] Per-category AUROC breakdown computed for all 13 OOD categories
- [ ] k-NN sensitivity re-run with ground truth labels
- [ ] Results JSON updated (label_source: "ground_truth")
- [ ] RESULTS.md rewritten without circular-evaluation caveats
- [ ] 4 figures regenerated + per-category heatmap added
- [ ] Clear verdict: does Mahalanobis still justify its complexity with fair evaluation?
- [ ] Optimal thresholds reported at 95% and 99% TPR for best method
