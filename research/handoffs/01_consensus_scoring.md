# Handoff: Multi-Model Consensus Scoring

**Priority**: High | **Effort**: Low | **Est. compute cost**: $0 (existing data)
**Addresses**: Paper 1, Paper 2, Paper 7 peer review items
**License**: CC BY-SA 4.0, Copyright 2025 Byron Williams

---

## Objective

Determine whether averaging predictions from 2-3 VLMs produces better pseudo-labels than any single model. This directly informs which teacher configuration the pseudo-labeling pipeline (Paper 7) should use.

## Hypothesis

Averaging Gemini 3 Flash + GPT-4.1 predictions yields wSRCC > 0.72 on DIQA-5000 test, exceeding either model alone. We also expect the consensus to be more robust on OOD (synthetic) data.

## Why This Matters

- If consensus beats single-model by even 0.01-0.02 wSRCC, it changes the entire pseudo-labeling strategy
- Addresses peer review items in 3 papers simultaneously
- Zero API cost — all data already collected

## Peer Review Items Addressed

| Paper | Item | Feedback |
|-------|------|----------|
| 2 | GPT-5.2, Grok | "Quantitatively test multi-model consensus (mean/median ensemble) on existing data" |
| 2 | High priority | "Ensemble VLM pseudo-labeling: GPT-4.1 + Gemini 3 Flash ensemble outperforms either alone on OOD" |
| 1 | Consensus (High) | "Consensus scoring (2-3 models): Averaging Gemini 3 Flash + GPT-4.1 predictions yields wSRCC > 0.72" |
| 7 | High priority | "Consensus vs. single-model pseudo-labels: Student trained on 2-model consensus labels outperforms single-model labels by >0.01 wSRCC" |

## Input Data

### DIQA-5000 Test Set (Real, ID)

**Ground truth**: `results/vlm_teacher_eval/full_eval/data/test.csv`
- 1,000 images, columns: `res`, `ori`, `overall`, `sharpness`, `color_fidelity`
- MOS scale: 1-5 (continuous)

**Checkpoint files** (9 models × 1,000 samples):
```
results/vlm_teacher_eval/full_eval/checkpoints/
├── google__gemini-3-flash-preview.jsonl
├── openai__gpt-4.1.jsonl
├── google__gemini-2.5-pro.jsonl
├── anthropic__claude-haiku-4.5.jsonl
├── qwen__qwen3-vl-8b-instruct.jsonl
├── qwen__qwen3-vl-8b-thinking.jsonl
├── qwen__qwen3-vl-8b-thinking__temp0.jsonl
├── qwen__qwen3.5-flash-02-23.jsonl
└── google__gemini-3-flash-preview__no_resize.jsonl
```

### Synthetic OOD Test Set

**Checkpoint files** (7 models × 520 samples):
```
results/vlm_teacher_eval/full_eval/checkpoints_synthetic/
├── google__gemini-3-flash-preview.jsonl
├── openai__gpt-4.1.jsonl
├── google__gemini-2.5-pro.jsonl
├── anthropic__claude-haiku-4.5.jsonl
├── qwen__qwen3-vl-8b-instruct.jsonl
├── qwen__qwen3-vl-8b-thinking.jsonl
└── qwen__qwen3.5-flash-02-23.jsonl
```

### Checkpoint Record Schema

```json
{
  "model_id": "qwen/qwen3.5-flash-02-23",
  "image": "test_res_00001.jpg",
  "overall": 4.5,
  "sharpness": 5.0,
  "color_fidelity": 4.5,
  "reasoning": "...",
  "latency_ms": 18624,
  "error": ""
}
```

Join key: `image` field → `res` column in test.csv.

## Existing Infrastructure

### Shared utilities (already built)

```python
from research.papers.shared.data_loader import (
    load_vlm_checkpoints,   # Load JSONL by model ID and split
    load_ground_truth,      # Load test.csv as dict
    merge_predictions_with_gt,  # Inner join on image name
    compute_metrics,        # SRCC, PLCC, MAE, RMSE, wSRCC with bootstrap CIs
)
from research.papers.shared.constants import (
    DIMENSIONS,        # ["overall", "sharpness", "color_fidelity"]
    WSRCC_WEIGHTS,     # [0.5, 0.25, 0.25]
    PRIMARY_MODELS,    # 7 primary model IDs
)
```

### Existing ensemble code (reference)

`research/correlation/ood_spread_analysis.py` already computes `ensemble_mean = df[model_columns].mean(axis=1).values` with SRCC/PLCC. Use as a starting pattern.

## Deliverables

### 1. Analysis script: `research/consensus/analyze_consensus.py`

Compute and compare the following configurations on both ID and OOD splits:

**Single models (baselines)**:
- Each of the 7 primary models individually

**Pairwise ensembles**:
- Gemini 3 Flash + GPT-4.1 (mean)
- Gemini 3 Flash + Gemini 2.5 Pro (mean)
- GPT-4.1 + Gemini 2.5 Pro (mean)

**Three-model ensembles**:
- Top-3 by wSRCC (mean)
- Top-3 by wSRCC (median)

**All-model ensemble**:
- All 7 primary models (mean)
- All 7 primary models (median)

**Weighted ensembles** (if simple mean wins):
- Inverse-MAE weighting
- wSRCC-proportional weighting

For each configuration, report per-dimension (SRCC, PLCC, MAE) + aggregate wSRCC with bootstrap 95% CIs.

### 2. Results file: `research/consensus/consensus_results.json`

Structured JSON with all metrics for every configuration.

### 3. Summary table: `research/consensus/RESULTS.md`

Markdown table ranking all configurations by wSRCC on ID, with OOD wSRCC alongside. Highlight the best configuration and the margin over best single model.

### 4. Figures: `research/consensus/figures/`

- `consensus_wsrcc_comparison.png` — Bar chart: single models vs ensemble configs, ID and OOD side by side
- `consensus_improvement_heatmap.png` — Heatmap: pairwise ensemble gain over each component model, per dimension

## Evaluation Criteria

The main question to answer: **Does any consensus configuration beat the best single model on wSRCC, and by how much?**

Secondary questions:
- Does consensus help more on OOD than ID?
- Is mean or median better?
- Do more models always help, or is there a sweet spot?
- Which dimensions benefit most from consensus?

## Technical Notes

- Filter out records where `error != ""` before computing metrics
- Use `compute_metrics` from shared infrastructure for consistent bootstrap CIs (n=1000, seed=42)
- wSRCC formula: `0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color`
- For OOD split, ground truth is the synthetic degradation parameter (not human MOS) — interpret correlations as "agreement with degradation parameters"
- The `no_resize` variant of Gemini 3 Flash is a prompt variant, not a separate model — exclude from consensus combinations unless explicitly testing prompt diversity

## Dependencies

```
numpy, scipy, pandas, matplotlib, json
```

All available in `DeQA-Score/.venv/`. Run with:
```bash
cd DeQA-Score && .venv/bin/python ../research/consensus/analyze_consensus.py
```

## Definition of Done

- [ ] All single-model baselines computed with bootstrap CIs
- [ ] All ensemble configurations computed with bootstrap CIs
- [ ] Best configuration identified with statistical significance assessment
- [ ] Results JSON written
- [ ] Summary markdown with ranked table written
- [ ] Two figures generated
- [ ] Clear recommendation for Paper 7 pipeline: which teacher config to use
