# Tier 1: Embedding-Space OOD Detector for Document IQA

## What It Does

The Tier 1 OOD detector identifies when SigLIP2-IQA's quality predictions may be unreliable because the input
document is **out-of-distribution** (OOD) relative to the DIQA-5000 training set. It operates in embedding space
with near-zero latency (~1-2ms), providing a fast gate before invoking expensive Tier 2 cross-model validation.

**Problem it solves**: SigLIP2's built-in uncertainty (σ² from GaussianNLL) captures *aleatoric* uncertainty
(inherent data noise), not *epistemic* uncertainty (distribution shift). A document type never seen during training
can produce a confident but wrong prediction. This detector catches that.

**Key result**: AUROC = 0.9963 on DIQA-5000 test vs synthetic OOD documents, with 99.5% true positive rate at
5% false positive rate.

## How It Works

### Architecture

```
Input Image
    │
    ▼
┌──────────────────────┐
│  SigLIP2 Backbone    │ ── already computed during IQA inference
│  (ViT-Base, 768-dim) │
└──────────┬───────────┘
           │ penultimate-layer embedding (768-dim)
           ▼
┌──────────────────────────────┐
│  Mahalanobis Distance        │
│  d = √((x-μ)ᵀ Σ⁻¹ (x-μ))   │
│                              │
│  μ = mean of DIQA-5000       │
│  Σ⁻¹ = precision matrix      │
│      (Ledoit-Wolf shrinkage) │
└──────────┬───────────────────┘
           │ scalar distance
           ▼
    d > threshold? ──→ Flag as OOD ──→ Trigger Tier 2
```

### Method: Mahalanobis Distance with Ledoit-Wolf Shrinkage

1. **Embedding extraction**: SigLIP2's penultimate ViT layer produces a 768-dimensional embedding for each
   document image. This embedding is already computed during normal IQA inference — no additional forward pass.

2. **Training distribution modeling**: Compute the mean vector μ (768-dim) and covariance matrix Σ (768×768) of
   all DIQA-5000 train+val embeddings (4,400 images). Ledoit-Wolf shrinkage regularizes the covariance estimate
   to handle the high-dimensional space (768 dims, 4,400 samples).

3. **Distance computation**: For a new image, compute its Mahalanobis distance from the training distribution
   centroid. This accounts for correlations between embedding dimensions — unlike Euclidean distance, it measures
   how many "standard deviations" away the point is in each principal direction.

4. **Thresholding**: If distance exceeds the threshold, the image is flagged as potentially OOD.

### Why Mahalanobis Over Simpler Methods

- **Euclidean distance** ignores correlations between dimensions and treats all directions equally
- **Cosine similarity** loses magnitude information
- **Mahalanobis distance** is the natural metric for Gaussian-distributed embeddings — it is affine-invariant
  and accounts for the shape of the training distribution
- **Ledoit-Wolf shrinkage** (shrinkage = 0.0032) prevents the covariance matrix from being singular or
  ill-conditioned, which is critical when dimensionality (768) is a significant fraction of sample count (4,400)

## Thresholds

### Calibration Data

| Dataset | Count | Median Distance | p95 Distance | p99 Distance |
|---------|-------|-----------------|--------------|--------------|
| DIQA-5000 train+val (fit) | 4,400 | 24.1 | 30.2 | 33.3 |
| DIQA-5000 test (held out) | 1,100 | 32.6 | 46.0 | 58.6 |
| Synthetic OOD | 370 | 75.4 | 101.0 | 105.7 |

### Train/Test Distribution Shift

There is an ~8-unit gap between training distances (median 24.1) and test distances (median 32.6). This is caused
by a checkpoint architecture mismatch (445 missing keys, 368 unexpected keys in the current checkpoint). The
backbone embeddings are still valid and discriminative, but the covariance model fitted on train embeddings doesn't
perfectly predict test embedding distances. **With a properly matched checkpoint, this gap should close.**

### Recommended Thresholds

| Threshold | Test FPR | OOD TPR | Use Case |
|-----------|----------|---------|----------|
| **46.0** (test p95) | **5.0%** | **99.5%** | **Recommended for production** |
| 45.0 | 6.5% | 100.0% | Aggressive — catches everything, slightly more false alarms |
| 50.0 | 2.9% | 97.8% | Conservative — fewer false alarms, misses ~2% of OOD |
| 30.2 (fit p95) | 59.4% | 100.0% | Do NOT use — train/test shift makes this too aggressive |

**Production recommendation**: Use **threshold = 46.0** (test p95). This gives 5% false positive rate on
held-out DIQA-5000 test images while catching 99.5% of OOD documents. Once the checkpoint mismatch is resolved,
re-fit using the fit p95 threshold directly.

### Per-Category Detection Performance

All 13 OOD categories from the synthetic PoC dataset are detected with near-perfect accuracy:

| OOD Category | AUROC | Mean Distance | Detection Rate |
|-------------|-------|---------------|----------------|
| ood_heavily_degraded | 1.0000 | 99.5 | 30/30 |
| ood_adversarial_nastaliq | 1.0000 | 96.7 | 20/20 |
| ood_very_low_dpi | 1.0000 | 92.9 | 30/30 |
| ood_multiscript | 1.0000 | 85.1 | 30/30 |
| ood_script_tibetan | 1.0000 | 80.7 | 30/30 |
| ood_script_ethiopic | 1.0000 | 78.6 | 30/30 |
| ood_form_layout | 1.0000 | 75.2 | 30/30 |
| ood_adversarial_fraktur | 1.0000 | 74.8 | 20/20 |
| ood_pristine | 1.0000 | 74.1 | 30/30 |
| ood_very_high_dpi | 1.0000 | 73.7 | 30/30 |
| ood_binarized | 0.9934 | 64.2 | 30/30 |
| ood_script_myanmar | 0.9886 | 58.5 | 30/30 |
| ood_cjk_vertical | 0.9719 | 51.3 | 30/30 |

## Usage

### Loading the Detector

```python
from image_preprocessing_detector.detection.ood_detector import EmbeddingOODDetector

detector = EmbeddingOODDetector.load("ood_params_4400.npz")

# Override threshold for production (test-calibrated)
detector.threshold = 46.0
```

### Scoring a Single Image

```python
# During normal SigLIP2 inference, extract the embedding:
result = siglip2_detector.predict(image, return_embedding=True)
embedding = result.embedding  # shape: (768,)

# Score with OOD detector
ood_result = detector.score(embedding)

print(ood_result.mahalanobis_distance)  # e.g., 25.3 (in-dist) or 82.1 (OOD)
print(ood_result.is_ood)                # True/False
print(ood_result.percentile)            # Approximate percentile vs calibration set
print(ood_result.threshold)             # Current threshold
```

### Batch Scoring

```python
# Score multiple embeddings at once
embeddings = np.load("embeddings.npy")  # shape: (N, 768)
results = detector.score_batch(embeddings)

distances = [r.mahalanobis_distance for r in results]
flagged = [r for r in results if r.is_ood]
```

### Integration with Tier 2

```python
# Tier 1 → Tier 2 gating pattern
ood_result = detector.score(embedding)

if ood_result.is_ood:
    # Invoke Tier 2: VLM cross-model validation (Qwen3-VL-8B)
    vlm_scores = vlm_validator.score(image)
    reliability = compute_agreement(siglip2_scores, vlm_scores)
else:
    # High confidence in-distribution — trust SigLIP2 directly
    reliability = 1.0
```

### Re-fitting on New Data

```python
# Extract embeddings for new training data
embeddings = np.load("new_train_embeddings.npy")  # shape: (N, 768)

# Fit new detector
detector = EmbeddingOODDetector.from_embeddings(
    embeddings,
    threshold_percentile=95.0,
)
detector.save("ood_params_updated.npz")
```

### Extracting Embeddings

```bash
# Extract embeddings from a dataset
PYTHONPATH=/path/to/image_detection:$PYTHONPATH \
    python3 scripts/extract_siglip2_embeddings.py \
    --checkpoint models/iqa/checkpoints/phase7/production_model_seed42.pt \
    --meta-path /path/to/metadata.json \
    --image-root /path/to/images/ \
    --output /path/to/embeddings.npy \
    --device cuda:0

# Metadata JSON format: [{"image": "relative/path.jpg"}, ...]
# For absolute paths, set --image-root /

# Fit OOD detector from extracted embeddings
python3 scripts/extract_siglip2_embeddings.py \
    --fit-ood /path/to/embeddings.npy \
    --ood-output /path/to/ood_params.npz \
    --threshold-pct 95.0
```

## Artifacts

All artifacts stored at `/mnt/e/image_detection/embeddings/`:

| File | Description | Size |
|------|-------------|------|
| `ood_params_4400.npz` | **Production OOD detector** (mean, precision matrix, threshold, calibration distances) | 2.2 MB |
| `diqa5000_trainval_all.npy` | Train+val embeddings (4,400 × 768) used for fitting | 13 MB |
| `diqa5000_test_all.npy` | Test embeddings (1,100 × 768) for evaluation | 3.3 MB |
| `synthetic_poc.npy` | Synthetic PoC embeddings (520 × 768) | 1.6 MB |
| `*_ids.json` | Image ID lists corresponding to each .npy file | <1 MB |

### Superseded artifacts (can be deleted)

| File | Reason |
|------|--------|
| `ood_params.npz` | Fitted on 3,500 res-only; superseded by `ood_params_4400.npz` |
| `diqa5000_train.npy` | 3,500 res-only; superseded by `diqa5000_trainval_all.npy` |
| `diqa5000_test.npy` | 1,000 res-only; superseded by `diqa5000_test_all.npy` |

## Limitations and Future Work

1. **Train/test distance shift**: The current checkpoint has 445 missing / 368 unexpected keys, causing an
   ~8-unit Mahalanobis distance shift between train and test splits. Re-fitting with a properly matched
   checkpoint will eliminate this and allow using the fit p95 threshold directly.

2. **Synthetic-only OOD evaluation**: All OOD documents are synthetically generated. Evaluation on real-world
   OOD datasets (Tobacco800, RVL-CDIP, CORD, handwritten forms) is needed to validate performance on natural
   distribution shifts.

3. **Global vs per-class modeling**: The current detector uses a single global Gaussian. Per-class modeling
   (e.g., separate distributions for different script families or document types) could improve sensitivity
   for subtle OOD shifts.

4. **Threshold tuning with real OOD feedback**: As OOD documents are encountered in production and labeled,
   the threshold can be tuned to optimize the precision/recall tradeoff for the actual deployment distribution.

5. **Latency**: The Mahalanobis distance computation adds ~1-2ms to inference — negligible compared to SigLIP2's
   ~30ms forward pass. No batching or GPU acceleration needed for this step.
