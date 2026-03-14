# Handoff: VLM Teacher Calibration on DIQA-5000

**Priority**: Critical | **Effort**: Low-Medium | **Est. compute cost**: ~$15-25 (API calls)
**Addresses**: Paper 7 blocking peer review items (unanimous)
**Depends on**: Handoff 01 (consensus scoring) for teacher selection
**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

---

## Objective

Fit per-model isotonic calibration curves for the top VLM teachers (Gemini 3 Flash, GPT-4.1) on the DIQA-5000 training split (3,500 images). This is the critical prerequisite for the first end-to-end pseudo-labeling expansion cycle.

## Why This Matters

Paper 7 received **Major Revision** — the only paper with that verdict. All 4 reviewers flagged this as a blocking issue:

> "VLM calibration demonstrated on SigLIP2 not VLM teachers — must fit per-model isotonic calibration on 3,500 training images" — 4/4 unanimous

The existing "14x MAE reduction" was demonstrated on SigLIP2 (the student model), not on the VLM teachers that will actually generate pseudo-labels. Without VLM-specific calibration:
- Pseudo-labels will have systematic bias (VLMs tend to over-rate quality)
- The pipeline's confidence weighting will be miscalibrated
- The end-to-end expansion cycle cannot produce trustworthy labels

## Peer Review Items Addressed

| Paper | Item | Feedback |
|-------|------|----------|
| 7 | 4/4 unanimous, **Critical** | "Calibration demonstrated on student model, not VLM teachers — must fit per-model isotonic calibration on 3,500 training images" |
| 7 | 3/4 majority, **Important** | "'14x MAE reduction' headline misleading — primarily scale mismatch [0,1] vs [1,5], not calibration difficulty" |
| 7 | GPT-5.2, **Minor** | "MainScore vs wSRCC metric confusion across tables — define formula at first use" |
| 7 | Gemini 3.1 Pro, **Suggestion** | "Add small calibration generalization test: human-labeled OOD subset with ID-fitted calibration curve" |
| 1 | Consensus, High | "Per-model calibration on 3,500 training images" |

## Approach

### Phase 1: Generate VLM predictions on training split (~$15-25)

Run Gemini 3 Flash and GPT-4.1 on all 3,500 DIQA-5000 training images using the same prompt template used for the test set evaluation. This produces per-image predictions for all 3 dimensions.

### Phase 2: Fit calibration curves ($0)

For each model × dimension (2 models × 3 dimensions = 6 calibration curves), fit:
- Linear regression (baseline)
- 4-parameter logistic (standard IQA practice)
- Isotonic regression (non-parametric, recommended)

### Phase 3: Evaluate on held-out data ($0)

Apply calibration curves fitted on training split to:
- DIQA-5000 test split (1,000 images) — existing predictions available
- Synthetic OOD split (520 images) — existing predictions available

## Input Data

### Training Ground Truth

```
DeQA-Score/Data-DeQA-Score/DIQA/metas/
├── train_diqa_overall.json      # 3,500 records
├── train_diqa_sharpness.json    # 3,500 records
└── train_diqa_color.json        # 3,500 records
```

Record schema:
```json
{
  "id": "image00004.jpg",
  "image": "DIQA/train/res/train_res_00004.jpg",
  "gt_score": 3.233,
  "gt_score_norm": 3.971,
  "level_probs": [0, 0.971, 0.029, 0, 0],
  "std": 0.8
}
```

Key fields: `id` (join key), `gt_score` (ground truth MOS on 1-5 scale).

### Test Set Predictions (already collected)

```
results/vlm_teacher_eval/full_eval/checkpoints/
├── google__gemini-3-flash-preview.jsonl    # 1,000 records
└── openai__gpt-4.1.jsonl                   # 1,000 records
```

### Test Ground Truth

```
results/vlm_teacher_eval/full_eval/data/test.csv
```

1,000 records with columns: `res`, `ori`, `overall`, `sharpness`, `color_fidelity`.

### Synthetic OOD Predictions (already collected)

```
results/vlm_teacher_eval/full_eval/checkpoints_synthetic/
├── google__gemini-3-flash-preview.jsonl    # 520 records
└── openai__gpt-4.1.jsonl                   # 520 records
```

### Reference: Existing SigLIP2 Calibration

The SigLIP2 calibration script and results serve as a template:
- Script: `results/siglip2_diqa5000/calibrate_isotonic.py`
- Results: `results/siglip2_diqa5000/calibration_results.json`

This script implements `fit_calibrators()`, `compute_dim_metrics()`, and `run_calibration()`. Adapt for VLM predictions.

### Prompt Template

The VLM evaluation prompt is in the evaluation scripts. Key details:
- VLMs output integer scores 1-5 per dimension (overall, sharpness, color_fidelity)
- Some responses need parsing from JSON or structured text
- Existing checkpoint files have already been parsed — `overall`, `sharpness`, `color_fidelity` fields are numeric

## Deliverables

### 1. VLM inference script: `research/vlm_calibration/run_vlm_training_eval.py`

Run Gemini 3 Flash and GPT-4.1 on 3,500 training images. Output format matches existing checkpoint schema:

```json
{
  "model_id": "google/gemini-3-flash-preview",
  "image": "train_res_00001.jpg",
  "overall": 4.0,
  "sharpness": 4.5,
  "color_fidelity": 4.0,
  "reasoning": "...",
  "latency_ms": 1234,
  "error": ""
}
```

Save to:
```
research/vlm_calibration/checkpoints/
├── google__gemini-3-flash-preview__train.jsonl
└── openai__gpt-4.1__train.jsonl
```

**Important**: Use the exact same prompt template, temperature, and parameters as the test set evaluation. Check existing evaluation scripts for the prompt:
- `research/ocr_iqa_correlation/scripts/06_vlm_eval.py` (may have prompt template)
- `results/vlm_teacher_eval/full_eval/` directory for evaluation config

**Cost estimate**: ~3,500 images × 2 models × ~$0.002-0.004/image ≈ $15-25 total.

### 2. Calibration fitting: `research/vlm_calibration/fit_calibration.py`

For each of the 6 model×dimension combinations:

```python
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import curve_fit

def fit_linear(x_train, y_train):
    """y = ax + b"""

def fit_logistic_4pl(x_train, y_train):
    """y = d + (a-d) / (1 + (x/c)^b) — standard IQA practice"""

def fit_isotonic(x_train, y_train):
    """Non-parametric monotonic mapping"""
```

Train on 3,500 training predictions vs ground truth MOS. Save fitted models as pickle or JSON.

### 3. Evaluation: `research/vlm_calibration/evaluate_calibration.py`

Apply each calibration method to test (1,000) and synthetic OOD (520) predictions. Report:

| Metric | Per dimension | Aggregate |
|--------|--------------|-----------|
| SRCC | ✓ (invariant under monotonic transforms) | wSRCC |
| PLCC | ✓ (should improve) | weighted |
| MAE | ✓ (primary calibration metric) | wMAE |
| RMSE | ✓ | weighted |
| Max absolute error | ✓ | - |

All with bootstrap 95% CIs (n=1000, seed=42).

### 4. Cross-domain generalization test

Apply calibration curves fitted on DIQA-5000 training to:
- DIQA-5000 test (same domain) — expected to work well
- Synthetic OOD (different domain) — the critical question

Report whether calibration curves generalize or whether per-domain calibration is needed.

### 5. Results: `research/vlm_calibration/calibration_results.json`

```json
{
  "gemini_3_flash": {
    "train_fit": {
      "overall": {"n": 3500, "raw_mae": ..., "isotonic_mae": ..., ...},
      "sharpness": {...},
      "color_fidelity": {...}
    },
    "test_eval": {
      "overall": {"raw_mae": ..., "isotonic_mae": ..., "raw_plcc": ..., "isotonic_plcc": ..., ...},
      ...
    },
    "ood_eval": {...}
  },
  "gpt_4_1": {...}
}
```

### 6. Summary: `research/vlm_calibration/RESULTS.md`

- Before/after calibration comparison table
- Scatter plots: raw vs calibrated predictions
- Recommendation: which calibration method to use in the pipeline

### 7. Figures: `research/vlm_calibration/figures/`

- `raw_vs_calibrated_scatter.png` — 2×3 grid (2 models × 3 dimensions): raw predictions vs GT and calibrated predictions vs GT
- `calibration_curves.png` — The fitted calibration functions overlaid on training data
- `mae_reduction_bar.png` — Before/after MAE by model and dimension
- `ood_generalization.png` — MAE on ID vs OOD for each calibration method

## Technical Notes

### VLM prediction scale

VLMs output integer-ish scores on a 1-5 scale (matching MOS). Unlike SigLIP2 which outputs on [0,1], VLMs are already on the right scale but have systematic bias:
- Over-rating: VLMs tend to rate higher than human MOS
- Compression: VLMs use a narrower range (e.g., 3-5 instead of 1-5)
- Dimension-specific bias: Color fidelity ratings may differ from overall

The "14x MAE reduction" seen with SigLIP2 was primarily scale correction ([0,1] → [1,5]). For VLMs, expect more modest but still meaningful MAE reduction from bias correction.

### Image paths for training

Training images are at paths like `DIQA/train/res/train_res_00004.jpg` relative to the data root. The full path would be something like `Data-DeQA-Score/DIQA/train/res/train_res_NNNNN.jpg`. Verify the exact image root before running inference.

### API rate limiting

At ~3,500 images × 2 models:
- Gemini 3 Flash: generous rate limits, should complete in ~1-2 hours
- GPT-4.1: may need throttling, ~2-3 hours
- Add retry logic with exponential backoff
- Save checkpoints every 100 images so failures are recoverable

### Matching image IDs

Training ground truth uses `id` field like `image00004.jpg`. Training images are named `train_res_00004.jpg`. The join logic needs to handle this mapping. Check the existing `calibrate_isotonic.py` for the exact alignment code — it already solves this problem for SigLIP2.

## Dependencies

### For Phase 1 (inference)

```
google-generativeai or openai SDK
PIL/Pillow (image loading)
```

API keys needed:
- `GEMINI_API_KEY` (in `.env`)
- `OPENROUTER_API_KEY` (in `.env`) — or direct OpenAI key

### For Phase 2-3 (calibration + evaluation)

```
numpy, scipy, scikit-learn, matplotlib, json
```

All in `DeQA-Score/.venv/`.

## Sequencing

1. **Wait for Handoff 01 results**: If consensus scoring shows that a 2-model ensemble is clearly superior, calibrate the ensemble output instead of individual models. If single model is best, proceed with that model only.
2. **Phase 1** (inference): ~$15-25, 2-4 hours wall time
3. **Phase 2** (calibration fitting): ~5 minutes compute
4. **Phase 3** (evaluation): ~5 minutes compute
5. **Update Paper 7**: Revise calibration section with real VLM results, address "14x MAE" framing

## Definition of Done

- [ ] VLM predictions collected for 3,500 training images (2 models × 3 dimensions)
- [ ] Calibration curves fitted (linear, 4PL, isotonic) for all 6 model×dimension combos
- [ ] Test set evaluation with before/after MAE, PLCC comparison
- [ ] OOD generalization test: do ID-fitted curves work on OOD?
- [ ] Results JSON written
- [ ] Summary markdown with recommendation
- [ ] 4 figures generated
- [ ] Clear answer: what calibration method and configuration should Paper 7's pipeline use?
- [ ] Paper 7 revision plan updated with concrete numbers to replace placeholder claims
