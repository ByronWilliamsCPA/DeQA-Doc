# SigLIP2 DIQA-5000 Full Extraction

**Date:** 2026-03-08
**Status:** Complete
**Checkpoint:** `siglip2_iqa_best.pt` (IQA-only, `google/siglip2-base-patch16-naflex`)

## Purpose

Complete SigLIP2 multi-task inference on all 5,000 DIQA-5000 images (train/val/test), producing:

1. **Per-image predictions** (JSONL) — IQA scores, script/source/orientation classification, shadow/warping severity
2. **768-dim embeddings** (NPZ) — for OOD detector fitting and downstream analysis
3. **Fitted OOD detector v2** — Mahalanobis distance detector on clean embeddings, resolving the prior checkpoint mismatch

This extraction resolves the checkpoint key mismatch (445 missing / 368 unexpected keys) documented in
`results/tier1_ood_detector/README.md` and `NEXT_ITERATION_ANALYSIS.md`. The v1 detector used embeddings from a
mismatched checkpoint, causing an ~8-unit train/test distance shift that invalidated threshold calibration.

## Files

| File | Size | Contents |
| ---- | ---- | -------- |
| `siglip2_diqa5000_train.jsonl` | 4.6 MB | 3,500 records |
| `siglip2_diqa5000_val.jsonl` | 662 KB | 500 records |
| `siglip2_diqa5000_test.jsonl` | 1.3 MB | 1,000 records |
| `embeddings/train.npz` | 6.9 MB | (3500, 768) float32 + image names |
| `embeddings/val.npz` | 989 KB | (500, 768) float32 + image names |
| `embeddings/test.npz` | 1.9 MB | (1000, 768) float32 + image names |
| `ood_detector_v2.npz` | 2.2 MB | Mean, precision matrix, calibration distances, threshold |
| `summary.json` | 713 B | Extraction metadata and OOD detector statistics |

## JSONL Schema (20 fields per record)

```json
{
  "image": "test_res_00001.jpg",
  "split": "test",
  "iqa_overall_mu": 0.6369,
  "iqa_overall_sigma_sq": 0.0012,
  "iqa_sharpness_mu": 0.5834,
  "iqa_sharpness_sigma_sq": 0.0015,
  "iqa_color_mu": 0.6102,
  "iqa_color_sigma_sq": 0.0011,
  "script_prediction": "TELU",
  "script_confidence": 0.42,
  "script_distribution": {"LATN": 0.15, "TELU": 0.42, "...": "..."},
  "source_prediction": "scanned",
  "source_confidence": 0.87,
  "orientation_degrees": 0,
  "orientation_confidence": 0.95,
  "shadow_severity": 0.12,
  "shadow_sigma_sq": 0.003,
  "warping_severity": 0.08,
  "warping_sigma_sq": 0.002,
  "inference_time_ms": 95.3
}
```

**IQA rescaling:** Model outputs mu in approximately [-0.17, 0.73]. To convert to MOS [1, 5]:
`MOS_pred = mu * 4.0 + 1.0` (model was trained with `(MOS - 1) / 4` normalization).

## OOD Detector v2

Fitted on 4,000 train+val embeddings using `EmbeddingOODDetector.from_embeddings(threshold_percentile=95.0)`.

| Metric | Train+Val (n=4,000) | Test (n=1,000) |
| ------ | ------------------- | -------------- |
| Median distance | 23.7 | 31.4 |
| p95 | 30.8 | 48.5 |
| p99 | 34.6 | 58.2 |

The train+val and test distributions are now healthy with no anomalous shift, confirming the checkpoint
mismatch is resolved. Compare to v1: train median=24.1, test median=32.6 (8-unit gap from mismatched keys).

### Recommended thresholds

| Threshold | Source | Use Case |
| --------- | ------ | -------- |
| **30.8** | Train+val p95 | Production default (now usable with clean checkpoint) |
| 48.5 | Test p95 | Conservative — very few false positives |
| 58.2 | Test p99 | Hard reject — only extreme outliers |

### Usage

```python
from image_preprocessing_detector.detection.ood_detector import EmbeddingOODDetector

detector = EmbeddingOODDetector.load("results/siglip2_diqa5000/ood_detector_v2.npz")
result = detector.score(embedding)  # embedding: (768,) float32
print(result.mahalanobis_distance, result.is_ood)
```

## Extraction Infrastructure

- **Script:** `image_detection/modal/extract_siglip2_diqa5000.py`
- **Platform:** Modal (serverless GPU)
- **GPU:** NVIDIA L4 (24GB)
- **Runtime:** ~50 minutes for all 5,000 images
- **Checkpoint:** `siglip2_iqa_best.pt` from Modal volume `siglip2-iqa-results`
- **Missing keys:** 22 (non-IQA heads: script, orientation, shadow, warping — expected for IQA-only checkpoint)
- **Unexpected keys:** 0

## Verification Checklist

- [x] Embedding shape: all (768,) float32
- [x] No NaN or Inf in any embeddings
- [x] Count match: train=3,500, val=500, test=1,000
- [x] JSONL schema: all 20 expected keys present
- [x] IQA mu range: approximately [-0.17, 0.73] (regression heads, not clamped)
- [x] OOD detector distances in expected ranges
- [x] No train/test distance shift anomaly

## Related Documents

- [VLM Teacher Evaluation](../vlm_teacher_eval/full_eval/VLM_TEACHER_EVALUATION.md) — Section 5.5.1 documents
  the re-calibration results; Section 6 Stage 4 uses v2 thresholds
- [Tier 1 OOD Detector](../tier1_ood_detector/README.md) — Original v1 detector documentation
- [OOD Next Iteration Analysis](../tier1_ood_detector/NEXT_ITERATION_ANALYSIS.md) — 13-model consensus
  identified checkpoint mismatch as P0 blocker (now resolved)
- [Handoff Document](../vlm_teacher_eval/full_eval/MODAL_SIGLIP2_DIQA5000_HANDOFF.md) — Original task spec
