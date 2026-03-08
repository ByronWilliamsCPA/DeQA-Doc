# Handoff: SigLIP2 Multi-Task Inference on Full DIQA-5000 Dataset

**Date:** 2026-03-07
**From:** DeQA-Doc VLM Teacher Evaluation
**To:** image_detection team
**Priority:** P0 — blocks OOD detector calibration and pseudo-labeling pipeline

---

## Objective

Run the SigLIP2 multi-task model on ALL 5,000 DIQA-5000 images (train + val + test)
to extract the complete set of outputs needed for the pseudo-labeling pipeline:

1. **768-dim embeddings** — required to fit the Mahalanobis OOD detector on
   train+val (n=4,000) and calibrate thresholds on test (n=1,000)
2. **IQA predictions with uncertainty** — mu and sigma_sq for overall, sharpness,
   color per image (validates the model's quality predictions)
3. **Classification outputs** — script detection (19-class), document source
   (3-class), orientation (4-class) for dataset enrichment
4. **Severity predictions** — shadow and warping scores for quality filtering

### Why This Matters

The OOD detector currently uses embeddings extracted from a checkpoint with 445
missing keys (architecture mismatch), causing an ~8-unit train/test distance shift.
A clean extraction from the correct checkpoint resolves this calibration issue and
produces reliable thresholds for the pseudo-labeling pipeline.

The per-image predictions also provide the baseline for comparing VLM pseudo-labels
against the fine-tuned model's own predictions — a key validation for the teacher
evaluation paper.

---

## Dataset

### Location on GCS

```
gs://image_detection_b/image-preprocessing-detector/datasets/diqa-5000/diqa-5000/
  train/
    train.csv                    # 3,500 rows (res, ori, overall, sharpness, color_fidelity)
    res/                         # 3,500 images (train_res_00001.jpg ... train_res_03500.jpg)
    ori/                         # 3,500 original images
  val/
    val.csv                      # 500 rows
    res/                         # 500 images (val_res_00001.jpg ... val_res_00500.jpg)
    ori/
  test/
    test.csv                     # 1,000 rows
    res/                         # 1,000 images (test_res_00001.jpg ... test_res_01000.jpg)
    ori/
```

**Also available as:** `gs://assured-oss-457903-diqa5000/diqa5000-test.tar.gz` (test split only)

### Ground Truth CSV Format

```csv
res,ori,overall,sharpness,color_fidelity
train_res_00001.jpg,train_ori_00001.jpg,3.187,3.613,3.307
```

- `res`: Filename of the distorted/captured image (use this for inference)
- `ori`: Filename of the original document
- `overall`, `sharpness`, `color_fidelity`: Human MOS [1.0, 5.0] from 15 annotators

### Split Sizes

| Split | Images | Purpose |
|-------|--------|---------|
| train | 3,500 | Fit OOD detector (Gaussian centroid + covariance) |
| val | 500 | Include in OOD detector fit (total: 4,000 for fitting) |
| test | 1,000 | Calibrate OOD thresholds (held out from fit) |
| **Total** | **5,000** | |

---

## Model to Run

### SigLIP2 Multi-Task Detector

**Production wrapper:** `image_detection/src/image_preprocessing_detector/detection/siglip2_multitask.py`

**Class:** `SigLIP2MultiTaskDetector`

**Backbone:** `google/siglip2-base-patch16-naflex` (86M params, ViT-B/16)

**Checkpoint:** `siglip2_iqa_best.pt` from Modal volume `dociq-checkpoints`

### Loading

```python
from image_preprocessing_detector.detection.siglip2_multitask import (
    SigLIP2MultiTaskDetector,
    MultiTaskPrediction,
)

detector = SigLIP2MultiTaskDetector(
    checkpoint_path="/checkpoints/siglip2_iqa_best.pt",
)
```

### Inference (per image)

```python
import cv2
import numpy as np

# Load image as BGR numpy array
image_bgr = cv2.imread(image_path)

# Run full multi-task inference with embedding extraction
result: MultiTaskPrediction = detector.predict(
    image_bgr,
    return_embedding=True,  # CRITICAL: must be True for OOD detector
)

# Extract all outputs
record = {
    # Identifiers
    "image": filename,
    "split": split_name,

    # IQA predictions (mu + uncertainty)
    "iqa_overall_mu": result.iqa_overall.mu,
    "iqa_overall_sigma_sq": result.iqa_overall.sigma_sq,
    "iqa_sharpness_mu": result.iqa_sharpness.mu,
    "iqa_sharpness_sigma_sq": result.iqa_sharpness.sigma_sq,
    "iqa_color_mu": result.iqa_color.mu,
    "iqa_color_sigma_sq": result.iqa_color.sigma_sq,

    # Classification predictions
    "script_prediction": result.script.predicted_class,        # e.g., "LATN"
    "script_confidence": result.script.confidence,
    "script_distribution": result.script.distribution,         # dict of 19 class probs
    "source_prediction": result.source.predicted_class,        # "scanned"/"camera"/"born_digital"
    "source_confidence": result.source.confidence,
    "orientation_prediction": result.orientation_degrees,       # 0/90/180/270
    "orientation_confidence": result.orientation.confidence,

    # Severity predictions
    "shadow_severity": result.shadow.value,
    "shadow_sigma_sq": result.shadow.sigma_sq,
    "warping_severity": result.warping.value,
    "warping_sigma_sq": result.warping.sigma_sq,

    # Embedding (768-dim float32)
    "embedding": result.embedding.tolist(),                    # list of 768 floats

    # Metadata
    "inference_time_ms": result.inference_time_ms,
    "device": result.device,
}
```

---

## Required Outputs

### Output 1: Per-Sample JSONL (all predictions)

One JSONL file per split with all fields above:

```
siglip2_diqa5000_train.jsonl   # 3,500 lines
siglip2_diqa5000_val.jsonl     # 500 lines
siglip2_diqa5000_test.jsonl    # 1,000 lines
```

Each line is a JSON object with ALL fields from the inference section above.

### Output 2: Embeddings NPZ (for OOD detector fitting)

Separate numpy archive for efficient OOD detector operations:

```python
import numpy as np

# Save embeddings per split
np.savez_compressed(
    "siglip2_diqa5000_embeddings.npz",
    train_embeddings=train_embeddings,      # shape (3500, 768), float32
    val_embeddings=val_embeddings,          # shape (500, 768), float32
    test_embeddings=test_embeddings,        # shape (1000, 768), float32
    train_images=train_image_names,         # shape (3500,), str
    val_images=val_image_names,             # shape (500,), str
    test_images=test_image_names,           # shape (1000,), str
)
```

**This is the critical output for OOD detector re-calibration.** After extraction:

```python
from image_preprocessing_detector.detection.ood_detector import EmbeddingOODDetector

# Fit on train+val
fit_embeddings = np.concatenate([train_embeddings, val_embeddings])  # (4000, 768)
ood_detector = EmbeddingOODDetector.from_embeddings(
    fit_embeddings,
    threshold_percentile=95.0,
)

# Calibrate on test
test_distances = [ood_detector.score(e).mahalanobis_distance for e in test_embeddings]
print(f"Test p50: {np.percentile(test_distances, 50):.1f}")
print(f"Test p95: {np.percentile(test_distances, 95):.1f}")
print(f"Test p99: {np.percentile(test_distances, 99):.1f}")

# Save fitted detector
ood_detector.save("ood_detector_v2.npz")
```

### Output 3: Summary Metrics JSON

Compute IQA correlation metrics against ground truth MOS for validation:

```json
{
  "checkpoint": "siglip2_iqa_best.pt",
  "model_id": "google/siglip2-base-patch16-naflex",
  "max_num_patches": 784,
  "timestamp": "2026-03-07T...",
  "splits": {
    "train": {
      "n": 3500,
      "overall_srcc": 0.XXX,
      "overall_plcc": 0.XXX,
      "overall_mae": 0.XXX,
      "sharpness_srcc": 0.XXX,
      "color_srcc": 0.XXX,
      "wsrcc": 0.XXX,
      "mean_inference_ms": 0.XXX,
      "script_accuracy": 0.XXX,
      "orientation_accuracy": 0.XXX
    },
    "val": { ... },
    "test": { ... }
  },
  "ood_detector": {
    "fit_n": 4000,
    "fit_splits": ["train", "val"],
    "ledoit_wolf_shrinkage": 0.XXXX,
    "train_val_median_distance": 0.XXX,
    "train_val_p95": 0.XXX,
    "train_val_p99": 0.XXX,
    "test_median_distance": 0.XXX,
    "test_p95": 0.XXX,
    "test_p99": 0.XXX
  }
}
```

---

## Important: Output Range Verification

### IQA Score Range

The SigLIP2 model's IQA heads output `mu` values. Before computing metrics:

1. Run inference on 5 test images with known MOS
2. Check if `mu` is in [0, 1] or [1, 5]
3. If [0, 1]: rescale as `mos_pred = mu * 4.0 + 1.0`
4. If [1, 5]: use directly

**Verify this BEFORE running the full 5,000 images.** The training script
(`modal/train_siglip2_iqa_v2.py`) defines the label normalization — check
whether MOS was normalized to [0, 1] during training.

### Embedding Dimensionality

The embedding should be exactly 768 dimensions (SigLIP2 ViT-B hidden size).
Verify on the first image: `assert result.embedding.shape == (768,)`.

---

## Modal App Structure

### Recommended Approach

Create `modal/extract_siglip2_diqa5000.py` in the `image_detection` repo:

```python
import modal
from modal.shared.constants import (
    checkpoint_volume,
    gcs_secret,
)

app = modal.App("siglip2-diqa5000-extraction")

# Volume for outputs
output_volume = modal.Volume.from_name(
    "siglip2-diqa5000-outputs",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "transformers>=4.40",
        "scipy",
        "numpy",
        "opencv-python-headless",
        "scikit-learn",       # For Ledoit-Wolf in OOD detector
        "google-cloud-storage",
    )
    .copy_local_dir("src", "/app/src")
)

@app.function(
    gpu="l4",               # 24GB, 3GB needed
    timeout=3600,
    memory=16384,           # 16GB RAM
    volumes={
        "/checkpoints": checkpoint_volume,
        "/outputs": output_volume,
    },
    secrets=[gcs_secret],
    image=image,
)
def extract_split(split: str) -> dict:
    """Extract all SigLIP2 outputs for one DIQA-5000 split.

    Args:
        split: "train", "val", or "test"

    Returns:
        Summary metrics dict.
    """
    import json
    import csv
    import cv2
    import numpy as np
    from pathlib import Path
    from scipy import stats

    # 1. Download split from GCS
    download_diqa_split(split)  # -> /data/diqa5000/{split}/

    # 2. Load model
    from image_preprocessing_detector.detection.siglip2_multitask import (
        SigLIP2MultiTaskDetector,
    )
    detector = SigLIP2MultiTaskDetector(
        checkpoint_path="/checkpoints/siglip2_iqa_best.pt",
    )

    # 3. Load ground truth
    split_dir = Path(f"/data/diqa5000/{split}")
    csv_path = split_dir / f"{split}.csv"
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row["res"]] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color_fidelity": float(row["color_fidelity"]),
            }

    # 4. Run inference
    results = []
    embeddings = []
    image_names = []
    res_dir = split_dir / "res"

    for img_name in sorted(gt.keys()):
        img_path = res_dir / img_name
        image_bgr = cv2.imread(str(img_path))
        pred = detector.predict(image_bgr, return_embedding=True)

        record = {
            "image": img_name,
            "split": split,
            "iqa_overall_mu": pred.iqa_overall.mu,
            "iqa_overall_sigma_sq": pred.iqa_overall.sigma_sq,
            "iqa_sharpness_mu": pred.iqa_sharpness.mu,
            "iqa_sharpness_sigma_sq": pred.iqa_sharpness.sigma_sq,
            "iqa_color_mu": pred.iqa_color.mu,
            "iqa_color_sigma_sq": pred.iqa_color.sigma_sq,
            "script_prediction": pred.script.predicted_class,
            "script_confidence": pred.script.confidence,
            "script_distribution": pred.script.distribution,
            "source_prediction": pred.source.predicted_class,
            "source_confidence": pred.source.confidence,
            "orientation_degrees": pred.orientation_degrees,
            "orientation_confidence": pred.orientation.confidence,
            "shadow_severity": pred.shadow.value,
            "shadow_sigma_sq": pred.shadow.sigma_sq,
            "warping_severity": pred.warping.value,
            "warping_sigma_sq": pred.warping.sigma_sq,
            "inference_time_ms": pred.inference_time_ms,
        }
        results.append(record)
        embeddings.append(pred.embedding)
        image_names.append(img_name)

    # 5. Save JSONL (without embeddings — too large for JSON)
    output_dir = Path("/outputs")
    jsonl_path = output_dir / f"siglip2_diqa5000_{split}.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # 6. Save embeddings as NPZ
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    npz_path = output_dir / f"siglip2_diqa5000_{split}_embeddings.npz"
    np.savez_compressed(
        npz_path,
        embeddings=embeddings_arr,
        image_names=np.array(image_names),
    )

    # 7. Compute summary metrics
    # ... (SRCC/PLCC/MAE against ground truth)

    output_volume.commit()
    return summary_metrics


@app.function(timeout=300)
def fit_ood_detector() -> dict:
    """Fit OOD detector on train+val embeddings, calibrate on test."""
    import numpy as np

    output_dir = Path("/outputs")

    train = np.load(output_dir / "siglip2_diqa5000_train_embeddings.npz")
    val = np.load(output_dir / "siglip2_diqa5000_val_embeddings.npz")
    test = np.load(output_dir / "siglip2_diqa5000_test_embeddings.npz")

    fit_embeddings = np.concatenate([
        train["embeddings"], val["embeddings"]
    ])  # (4000, 768)

    from image_preprocessing_detector.detection.ood_detector import (
        EmbeddingOODDetector,
    )
    detector = EmbeddingOODDetector.from_embeddings(
        fit_embeddings, threshold_percentile=95.0,
    )

    # Calibrate on test
    test_distances = np.array([
        detector.score(e).mahalanobis_distance
        for e in test["embeddings"]
    ])

    detector.save(str(output_dir / "ood_detector_v2.npz"))
    output_volume.commit()

    return {
        "fit_n": len(fit_embeddings),
        "test_median": float(np.median(test_distances)),
        "test_p95": float(np.percentile(test_distances, 95)),
        "test_p99": float(np.percentile(test_distances, 99)),
    }


@app.local_entrypoint()
def main():
    # Run all splits (can parallelize train/val/test)
    for split in ["train", "val", "test"]:
        metrics = extract_split.remote(split)
        print(f"{split}: {metrics}")

    # Fit OOD detector after all extractions complete
    ood_result = fit_ood_detector.remote()
    print(f"OOD detector: {ood_result}")
```

---

## GPU and Cost Estimates

| Split | Images | Est. Time (L4) | Est. Time (T4) |
|-------|--------|-----------------|-----------------|
| train | 3,500 | ~6 min | ~12 min |
| val | 500 | ~1 min | ~2 min |
| test | 1,000 | ~2 min | ~4 min |
| OOD fit | — | ~10 sec | ~10 sec |
| **Total** | **5,000** | **~10 min** | **~20 min** |

| Config | GPU | Cost |
|--------|-----|------|
| Sequential (L4) | L4 24GB | ~$0.10 |
| Sequential (T4) | T4 16GB | ~$0.12 |
| Parallel 3 splits (L4) | 3x L4 | ~$0.12 (faster) |

**Recommended:** L4, sequential — simple and cheap.

---

## Output File Summary

Deliver these files back to the `DeQA-Doc` repo:

| File | Size Est. | Contents |
|------|-----------|----------|
| `siglip2_diqa5000_train.jsonl` | ~5MB | 3,500 records (all predictions, no embeddings) |
| `siglip2_diqa5000_val.jsonl` | ~700KB | 500 records |
| `siglip2_diqa5000_test.jsonl` | ~1.5MB | 1,000 records |
| `siglip2_diqa5000_train_embeddings.npz` | ~10MB | (3500, 768) float32 array |
| `siglip2_diqa5000_val_embeddings.npz` | ~1.5MB | (500, 768) float32 array |
| `siglip2_diqa5000_test_embeddings.npz` | ~3MB | (1000, 768) float32 array |
| `ood_detector_v2.npz` | ~4.5MB | Fitted detector (mean, precision matrix, calibration) |
| `siglip2_diqa5000_summary.json` | ~2KB | Metrics per split |

**Total output:** ~25MB

### Where to save in DeQA-Doc

```
DeQA-Doc/results/siglip2_diqa5000/
  siglip2_diqa5000_train.jsonl
  siglip2_diqa5000_val.jsonl
  siglip2_diqa5000_test.jsonl
  embeddings/
    train.npz
    val.npz
    test.npz
  ood_detector_v2.npz
  summary.json
```

---

## Verification Checklist

Run these checks before considering the extraction complete:

- [ ] **Embedding shape**: All embeddings are exactly (768,) float32
- [ ] **IQA range sanity**: mu values checked against known MOS on 5 test images
- [ ] **No NaN/Inf**: `assert not np.any(np.isnan(embeddings))` for all splits
- [ ] **Count match**: train=3,500, val=500, test=1,000 records
- [ ] **Script distribution**: Verify majority LATN for standard docs (spot check)
- [ ] **Orientation**: Verify 0 degrees for non-rotated images (spot check)
- [ ] **OOD detector fit**: Ledoit-Wolf shrinkage coefficient is small (< 0.01)
- [ ] **Test distance distribution**: Median ~25-35, p95 ~40-50, p99 ~55-65
  (if significantly different from these ranges, investigate checkpoint mismatch)
- [ ] **IQA metrics on test**: Overall SRCC should be ~0.89 (matching benchmark)
- [ ] **File sizes**: Embeddings NPZ ~10MB for train, ~3MB for test

---

## Dependencies

```
torch>=2.1.0
transformers>=4.40
opencv-python-headless
scipy
numpy
scikit-learn>=1.3        # For Ledoit-Wolf covariance
google-cloud-storage     # For GCS dataset download
```

The `image_preprocessing_detector` package must be importable (add `src/` to path
or install via pip).

---

## Contact

Questions about OOD detector methodology: see `results/tier1_ood_detector/README.md`
in DeQA-Doc repo.

Questions about embedding usage in pseudo-labeling pipeline: see
`results/vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md`, Section 5.5 and 6.
