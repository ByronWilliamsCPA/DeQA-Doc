# Handoff: k-NN OOD Detection Baseline Comparison

**Priority**: High | **Effort**: Low | **Est. compute cost**: $0 (existing embeddings)
**Addresses**: Paper 4 peer review items (unanimous)
**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

---

## Objective

Implement k-NN distance as an OOD detection baseline and compare against the existing Mahalanobis detector on identical SigLIP2 embeddings. This addresses the strongest criticism of Paper 4: no baseline OOD method comparison.

## Hypothesis

k-NN distance achieves comparable AUROC (within 0.01) to Mahalanobis without requiring covariance estimation, making it a simpler alternative for the pipeline.

## Why This Matters

- Paper 4's peer review unanimously flagged "no baseline OOD method comparisons (cosine, KNN, energy, GMM, one-class SVM) on same embeddings" as a blocking gap
- Validates that Mahalanobis is the right choice — or reveals a simpler alternative
- k-NN is the highest-priority experiment listed in Paper 4's RESEARCH_AGENDA.md
- Results directly feed Paper 7's gating design decisions

## Peer Review Items Addressed

| Paper | Item | Feedback |
|-------|------|----------|
| 4 | 5-model consensus (unanimous) | "No baseline OOD method comparisons (cosine, KNN, energy, GMM, one-class SVM) on same embeddings" |
| 4 | Gemini 3.1 Pro | "Single global Gaussian assumption may mask subtle OOD shifts as pipeline scales" |
| 4 | RESEARCH_AGENDA | "k-NN distance matches Mahalanobis AUROC without covariance estimation — Priority: High" |
| 7 | 4-model consensus | "Synthetic-only OOD evaluation — must validate on real-world datasets" (partial — this validates the method, real-world data is separate) |

## Input Data

### Embeddings (pre-extracted SigLIP2-Base, 768-dim)

```
results/siglip2_diqa5000/embeddings/
├── train.npz   # (3500, 768) float32 + image_names
├── val.npz     # (500, 768) float32 + image_names
└── test.npz    # (1000, 768) float32 + image_names
```

Each NPZ contains:
- `embeddings`: shape `(N, 768)`, dtype `float32`
- `image_names`: shape `(N,)`, dtype unicode strings

### Pre-fitted Mahalanobis Detector

```
results/siglip2_diqa5000/ood_detector_v2.npz
```

Contains:
- `mean`: `(768,)` float64 — centroid of train+val
- `precision_matrix`: `(768, 768)` float64 — inverse covariance (Ledoit-Wolf shrinkage)
- `calibration_distances`: `(4000,)` float32 — pre-computed Mahalanobis distances for train+val
- `threshold`: scalar float64 — default 46.0

### OOD Labels

The test set (1000 images) includes both ID and OOD samples. The existing detector metadata reports 536/1000 as OOD at the test p95 threshold. For ground truth OOD labels, use the synthetic checkpoint files:

```
results/vlm_teacher_eval/full_eval/checkpoints_synthetic/*.jsonl
```

These contain 520 synthetic OOD images. Any test image NOT in the synthetic set is ID. Alternatively, use the Mahalanobis calibration distances from train+val as the reference distribution and the test distances as the evaluation distribution.

**Important**: Confirm the exact OOD ground truth labeling convention with the project lead before computing AUROC. The summary.json reports thresholds but not per-image binary labels.

### Existing Mahalanobis Interface

```python
# DeQA-Score/src/uncertainty/ood_wrapper.py
from src.uncertainty.ood_wrapper import OODDetectorWrapper

detector = OODDetectorWrapper.from_npz(
    "results/siglip2_diqa5000/ood_detector_v2.npz",
    threshold=46.0
)
result = detector.score(embedding)       # Single: OODResult(distance, is_ood, threshold)
results = detector.score_batch(embeddings)  # Batch: List[OODResult]
```

### Summary Metadata

`results/siglip2_diqa5000/summary.json` — key thresholds:
- Train/val p95: 30.80
- Train/val p99: 34.60
- Test p95: 48.45 (5% FPR)
- Test p99: 58.22

## Deliverables

### 1. Detection methods: `research/ood_baselines/ood_methods.py`

Implement 4 OOD scoring functions, all operating on the same embeddings:

```python
def mahalanobis_scores(train_emb, test_emb) -> np.ndarray:
    """Existing method. Recompute from scratch for fair comparison."""

def knn_scores(train_emb, test_emb, k=10) -> np.ndarray:
    """k-NN distance: mean distance to k nearest train neighbors."""

def cosine_scores(train_emb, test_emb) -> np.ndarray:
    """1 - max cosine similarity to any train sample."""

def energy_scores(train_emb, test_emb) -> np.ndarray:
    """Negative log-sum-exp of cosine similarities (energy-based)."""
```

Each returns an array of shape `(N_test,)` where higher = more OOD.

### 2. Evaluation script: `research/ood_baselines/evaluate_ood.py`

For each method, compute:
- **AUROC** with bootstrap 95% CI (n=1000)
- **AUPRC** (precision-recall, since classes may be imbalanced)
- **FPR@95TPR** (false positive rate at 95% true positive rate)
- **FPR@99TPR**
- Per-category AUROC (if synthetic OOD category labels are available from the checkpoint metadata)
- ROC curves for overlay plot

### 3. k parameter sensitivity: `research/ood_baselines/knn_sensitivity.py`

Sweep k ∈ {1, 3, 5, 10, 20, 50, 100} and report AUROC for each. Identify optimal k.

### 4. Results: `research/ood_baselines/ood_baseline_results.json`

```json
{
  "methods": {
    "mahalanobis": {"auroc": 0.996, "auroc_ci": [0.993, 0.998], "fpr95": 0.05, ...},
    "knn_k10": {"auroc": ..., "auroc_ci": [...], "fpr95": ..., ...},
    "cosine": {...},
    "energy": {...}
  },
  "knn_sensitivity": {"k1": ..., "k3": ..., ...},
  "per_category": {...}
}
```

### 5. Summary: `research/ood_baselines/RESULTS.md`

Table comparing all methods. Clear recommendation for Paper 4 and Paper 7.

### 6. Figures: `research/ood_baselines/figures/`

- `roc_comparison.png` — Overlaid ROC curves for all 4 methods
- `auroc_bar_comparison.png` — Bar chart with CIs
- `knn_k_sensitivity.png` — AUROC vs k line plot
- `per_category_heatmap.png` — Method × OOD category AUROC heatmap (if category labels available)

## Technical Notes

### OOD ground truth construction

The train+val sets (4000 images) are all ID (DIQA-5000). The test set has 1000 DIQA-5000 images (ID). The synthetic set has 520 images (OOD). To construct binary labels:

**Option A** (if synthetic embeddings are separately available): Concatenate test ID + synthetic OOD embeddings, label accordingly.

**Option B** (if only test.npz is available): Use the existing Mahalanobis calibration distances as a proxy. This is circular for Mahalanobis evaluation but valid for comparing methods.

Check whether synthetic OOD embeddings exist somewhere in the repo. If not, they may need to be extracted from the synthetic images using SigLIP2.

### Reference distribution

For all methods, use train+val (4000 samples) as the reference "in-distribution" set. Load both `train.npz` and `val.npz` and concatenate.

### Computational considerations

- k-NN with k=10 on 4000 reference × 1000 test × 768 dims is fast (~seconds with scipy.spatial.KDTree or sklearn.neighbors)
- Cosine similarity matrix (4000 × 1000) fits in memory (~30 MB float32)
- No GPU needed

### Import path

The `src.uncertainty` module requires PYTHONPATH setup and has a CUDA-dependent `__init__.py`. To avoid import issues:

```python
import sys
sys.path.insert(0, "DeQA-Score")
# Import ood_wrapper directly, not through src package
from src.uncertainty.ood_wrapper import OODDetectorWrapper
```

Or simply reimplement Mahalanobis from scratch using the NPZ arrays (mean, precision_matrix) — it's ~5 lines of numpy.

## Dependencies

```
numpy, scipy, scikit-learn, matplotlib, json
```

All available in `DeQA-Score/.venv/`. Run with:
```bash
cd DeQA-Score && .venv/bin/python ../research/ood_baselines/evaluate_ood.py
```

## Definition of Done

- [ ] All 4 OOD scoring methods implemented
- [ ] AUROC, AUPRC, FPR@95TPR computed for all methods with bootstrap CIs
- [ ] k-NN sensitivity sweep completed
- [ ] Per-category breakdown computed (if category labels available)
- [ ] Results JSON written
- [ ] Summary markdown with recommendation written
- [ ] 3-4 figures generated
- [ ] Clear verdict: does Mahalanobis justify its complexity over k-NN?
