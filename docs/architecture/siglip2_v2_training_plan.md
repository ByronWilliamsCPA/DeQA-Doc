# SigLIP2-IQA v2.0 Training Plan

## Phase 1 Results (2026-03-11)

| Run | Patches | Attn Pool | Scheduler | PCGrad | Val wSRCC | Notes |
|-----|---------|-----------|-----------|--------|-----------|-------|
| A1  | 784     | off       | cosine    | off    | **0.8935** | Resolution-only baseline |
| A2  | 784     | **on**    | cosine    | off    | 0.8925    | Attention pooling = no gain |
| A3  | —       | —         | —         | —      | skipped   | A1/A2 delta too small to justify |
| A4  | —       | —         | —         | —      | skipped   | A1/A2 delta too small to justify |

**Conclusion**: Architecture ablation shows ~0.89 data ceiling. Both A1 and A2 converge to the same wSRCC, confirming Paper 10's prediction that **data is the bottleneck**, not architecture. Proceeding to Phase 2 (pseudo-label validation).

---

## Context

SigLIP2-IQA v1.0 achieves wSRCC=0.891 on DIQA-5000 test (86M params, ~100ms inference), outperforming all single models including 7B+ MLLMs. The v2.0 infrastructure (training script, model, data pipeline, PCGrad, configs) is fully implemented in `modal/` but has never been trained. The goal is to push toward 0.92+ wSRCC on DIQA-5000 test, then improve OOD generalization to diverse document types. Three data strategies are on the table: DIQA-only refinement, pseudo-label integration, or dataset expansion.

### Key Insight from Paper 10 (Backbone Probe)

Linear probing of 11 SigLIP2 variants reveals **resolution is the dominant factor** for document quality representation, especially sharpness:

- **Large-p16-512** (303M, 512px fixed) is the best backbone: wSRCC=0.822, sharpness SRCC=0.805
- **Base-p16-512** (86M, 512px fixed) captures 68% of the Large-512 advantage over Base-NaFlex — just from resolution
- **Scaling beyond Large hurts**: Giant-opt (1B) ranks 7th of 11; So400m NaFlex ranks 9th
- **NaFlex is scale-dependent**: helps at Base (+1.4%) but hurts at So400m (-2.0%)
- **Sharpness bottleneck is resolution, not capacity**: Base-512 (0.796 sharpness) > Giant-384 (0.757) despite 12x fewer params

This means our v2.0 plan should incorporate a **resolution increase experiment** (train Base at higher effective resolution via 784 patches) and evaluate whether a **Large-512 teacher** adds value for sharpness-targeted distillation.

**Recommendation: Stage A first (DIQA-only), then Stage B (pseudo-labels) if needed. Defer Stage C (external data) until OOD phase.**

---

## Phase 0: Pre-Training Agent Analysis

**Before any training runs**, the agent team evaluates the process design and baseline data.

### Agent 2 (Class Imbalance & Error Analyst) — Pre-Training

Analyze v1.0 predictions on DIQA-5000 test to determine if class imbalance is a problem:

1. Load `results/siglip2_diqa5000/siglip2_diqa5000_test.jsonl` + GT labels from `Data-DeQA-Score/DIQA/metas/`
2. Compute per-bin SRCC (excellent/good/fair/poor/bad) for each dimension
3. Identify the 50 worst-predicted test images — cluster by doc type, script, quality bin
4. Determine if the sharpness bottleneck correlates with specific quality bins or document types
5. **Recommendation**: Should we add inverse-frequency sampling/loss reweighting? Or is the imbalance not the issue?

### Agent 1 (Hyperparameter Sensitivity) — Pre-Training

Review the v2.0 configs and training script for potential issues:

1. Validate that the config parameter ranges are reasonable (LR, weight decay, batch size)
2. Check if 784 patches + batch_size=4 fits in A10 24GB (reference `modal/validate_siglip2_v2.py` results)
3. Estimate wall-clock time per run
4. Flag any training instability risks (PCGrad + warm restarts interaction, attention pooling initialization)

---

## Phase 1: DIQA-Only Ablation Sweep (Stage A)

**Why**: Isolate architectural gains before introducing data complexity. The Tier 1 improvements are bundled in the current config — we need to know which ones actually help.

**Execution**: Sequential runs (one at a time). Agent team analyzes results after each run, enabling mid-course correction before the next.

### Experiment 1.1: Tier 1 Ablation (4 runs, ~$60)

Run order chosen to build understanding incrementally:

| Order | Run | Patches | Attn Pool | Scheduler | PCGrad | Purpose |
|-------|-----|---------|-----------|-----------|--------|---------|
| 1st | A1 | 784 | off | cosine (no restarts) | off | Resolution-only baseline (Paper 10 says resolution is dominant) |
| 2nd | A2 | 784 | **on** | cosine (no restarts) | off | + attention pooling |
| 3rd | A3 | 784 | on | **warm restarts** | off | + scheduler |
| 4th | A4 | 784 | on | warm restarts | **on** | Full Tier 1 (current config) |

**After each run**: Agent team reviews results — if A1 already shows strong gains from 784 patches (Paper 10 predicts this), we may skip A2-A4 or reprioritize. If attention pooling hurts (A2 < A1), we drop it and skip A3/A4 as currently designed.

**Files to create**: `modal/configs/siglip2_v2_ablation_{a1,a2,a3,a4}.yaml` — fork from `siglip2_v2_diqa_only.yaml`, toggle `use_attention_pooling`, `phase2_scheduler`, `phase2_use_pcgrad`.

**Paper 10 integration**: A1 tests the core Paper 10 finding (resolution is the dominant variable). v1.0 trained at 576 patches (~384px effective), A1 trains at 784 patches (~448px effective). Paper 10 shows Base-512 linear probe at 0.812 vs Base-NaFlex at 0.793 (+0.019), so we expect a meaningful sharpness improvement from resolution alone.

### Experiment 1.2: Dropout Tuning (2 runs, ~$30)

Using the best config from 1.1:

| Run | Dropout | Rationale |
|-----|---------|-----------|
| A5 | 0.1 | Current 0.3 may be too aggressive for 3,500 samples |
| A6 | 0.15 | Moderate reduction |

### Experiment 1.3: Sharpness-Targeted (conditional, 1-2 runs, ~$30)

Only if sharpness remains the bottleneck after 1.1-1.2. Paper 10 suggests resolution should address most of the sharpness gap, so these may not be needed:

- **Asymmetric loss weights**: `{"overall": 1.0, "sharpness": 1.5, "color": 1.0}` — requires adding `dim_loss_weights` config field to `SigLIP2V2Config` and training script
- **Wider sharpness head**: `head_hidden: 512` for sharpness only — requires `per_dim_head_hidden` override in config

### Experiment 1.4: Teacher-Student Distillation (conditional, ~$20-30)

If Phase 1 wSRCC plateaus below 0.92 and Paper 10's Large-512 teacher gap is promising:

1. Fine-tune `google/siglip2-large-patch16-512` (303M) on DIQA-5000 (~$20, fits on A10)
2. Use the fine-tuned Large-512 as a teacher to generate soft labels for the Base student
3. Retrain Base student with teacher soft-label distillation loss
4. Paper 10 predicts +3.2% sharpness SRCC from the Large-512 teacher representation

This is higher effort and only justified if Experiments 1.1-1.3 leave a sharpness gap.

### Agent Analysis After Each Run

After each training run completes:

**Agent 1 (Hyperparameter Sensitivity)**:

- Compare this run's wSRCC to previous runs with bootstrap CIs
- Assess statistical significance of the improvement/regression
- Check for overfitting (train/val loss divergence)
- Log PCGrad conflicts (A4), attention entropy (A2-A4)

**Agent 2 (Error Analyst)**:

- Per-bin error analysis on the new checkpoint
- Did the resolution increase help specific quality bins?
- Did sharpness improve as Paper 10 predicted?
- Recommend whether to proceed with the next ablation or pivot

### Metrics to Track (all runs)

- Per-dimension SRCC, PLCC, MAE, RMSE
- wSRCC (composite)
- Train/val loss gap (overfitting indicator)
- PCGrad conflict counts (when enabled)
- Attention entropy per dimension (when attention pooling enabled)
- Sigma_sq distribution per dimension

### Decision Gate

| Phase 1 Result | Next Step |
|----------------|-----------|
| wSRCC >= 0.92 | Skip to Phase 3 (OOD) |
| wSRCC 0.90-0.92 | Phase 2 (pseudo-labels) or 1.4 (teacher-student) |
| wSRCC < 0.90 | Error analysis + Phase 0 class imbalance findings inform pivot |

---

## Phase 2: Pseudo-Label Integration (Stage B)

### Step 2.1: Validate Pseudo-Label Quality (~$0, local)

Run the pseudo-labeling pipeline on the 500 DIQA **val** images (which have GT for overall, sharpness, color) to measure pipeline accuracy:

1. Use existing v1.0 predictions from `results/siglip2_diqa5000/siglip2_diqa5000_val.jsonl`
2. Load val embeddings from `results/siglip2_diqa5000/embeddings/val.npz`
3. Process through `src/uncertainty/pseudo_label.py` pipeline (OOD + fusion, no DeQA cross-validator)
4. Convert SigLIP2 mu from [0,1] to MOS [1,5] via `MOS = 1 + 4 * mu`
5. Compare pseudo-label MOS to GT MOS — compute MAE, acceptance rate per tier, per dimension
6. This is a validation experiment, not training

**Note**: DIQA test set has no GT scores (challenge held-out). Val set (500 images) is the largest labeled set available for validation.

### Step 2.2: Controlled Mixing Experiments — DEFERRED

Skipped for now. Will revisit after evaluating pseudo-label quality from 2.1.

### Step 2.3: VLM Consensus Labels — PARTIALLY EXPLORED

26+ VLM evaluations already exist on DIQA test images. Initial investigation showed:

- **Isotonic regression calibration is NOT beneficial** — evaluated and determined not to improve VLM predictions
- Raw VLM consensus (median of top models) may still have value as external signal
- Top VLMs: Gemini 3 Flash, GPT-4.1, Qwen 3.5 122B

Remaining potential: use uncalibrated VLM consensus as "silver standard" labels (weight 0.3-0.5) to break circular training. Deferred until pseudo-label quality from 2.1 is assessed.

---

## Phase 3: OOD Generalization

Only after DIQA target is met. **BLOCKED**: waiting for external datasets to become available.

### Step 3.1: Real-World OOD Evaluation

1. Extract v2.0 embeddings from RVL-CDIP (sample 5,000), Tobacco800 (1,600)
2. Compute Mahalanobis distances — are these truly OOD?
3. Run v2.0 predictions + VLM consensus as approximate GT
4. Measure correlation — how well does the model generalize?

### Step 3.2: Controlled Degradation Pipeline

Apply synthetic degradations (blur, JPEG compression, noise, resolution reduction) to RVL-CDIP documents. Degradation parameters provide known quality labels without human annotation. Train on DIQA + degraded docs, measure:

- DIQA test wSRCC maintained? (catastrophic forgetting check)
- OOD correlation improved?

### Step 3.3: KONIQ-10k (low priority)

KONIQ is natural image IQA — only overall dimension maps to DIQA. High risk of negative transfer for sharpness/color. Run isolated experiment with KONIQ mapped to overall-only; monitor for sharpness/color degradation.

---

## Agent Team Assignments

Agents analyze the process design **before** training begins (Phase 0) and provide feedback **after each sequential run** to enable mid-course correction.

### Agent 1: Hyperparameter Sensitivity Analyst

**Phase 0 (pre-training)**:

- Review v2.0 configs for parameter reasonability (LR, weight decay, batch size)
- Validate VRAM fit at 784 patches (reference validate script results)
- Flag training instability risks (PCGrad + warm restarts interaction, attention pooling init)
- Estimate wall-clock time per run

**After each run**:

- Compare wSRCC to previous runs with bootstrap CIs — is the delta statistically significant?
- Check overfitting indicators (train/val loss divergence, sigma_sq trends)
- PCGrad conflict frequency analysis (A4 only)
- Attention entropy correlation with per-dimension SRCC (A2-A4)
- Recommend whether to proceed, pivot, or skip subsequent ablation runs

### Agent 2: Class Imbalance & Error Analyst

**Phase 0 (pre-training)** — determines if class imbalance needs addressing:

- Analyze v1.0 predictions by quality bin (excellent/good/fair/poor/bad) per dimension
- Identify the 50 worst-predicted test images — cluster by doc type, script, quality level
- Determine if sharpness bottleneck correlates with specific bins or document types
- **Deliver recommendation**: inverse-frequency sampling/loss reweighting, or skip

**After each run**:

- Per-bin error analysis on new checkpoint — did resolution increase help specific bins?
- Did sharpness improve as Paper 10 predicted?
- Compare error patterns across runs to identify systematic failures

### Agent 3: Pseudo-Label Quality Auditor

**Activates in Phase 2 only**:

- Run Step 2.1 validation: MAE between pseudo-labels and GT per dimension
- Acceptance rate per tier, percentage within 0.5/1.0 MOS of GT
- Systematic bias analysis by quality level and dimension
- VLM consensus vs. SigLIP2-alone correlation comparison

### Agent 4: OOD Generalization Evaluator

**Activates in Phase 3 only**:

- Mahalanobis distance distributions on RVL-CDIP vs. DIQA
- SigLIP2 prediction reliability on OOD documents (VLM consensus correlation)
- Catastrophic forgetting measurement (pre/post DIQA wSRCC with OOD data)

### Agent 5: Backbone & Teacher Analyst (NEW — informed by Paper 10)

**Phase 0 (pre-training)**:

- Review Paper 10 probe results to quantify expected gains from 784 patches
- Assess whether Large-512 teacher distillation is worth pursuing (cost vs. expected sharpness gain)
- Compare teacher-student vs. data expansion ROI based on probe data

**After Phase 1 (conditional)**:

- If sharpness gap persists, design the Large-512 fine-tuning experiment
- Estimate distillation label quality from probe SRCC gap

---

## Code Changes Required

### `modal/siglip2_v2_model.py` — Config additions

- `dim_loss_weights: dict[str, float]` (default `{"overall": 1.0, "sharpness": 1.0, "color": 1.0}`)
- `per_dim_head_hidden: dict[str, int] | None` (optional per-dimension override)
- `early_stopping_patience: int` (default 10)

### `modal/train_siglip2_iqa_v2.py` — Training loop

- Apply `dim_loss_weights` in loss computation
- Add early stopping on wSRCC plateau
- Add sample-level confidence weighting for pseudo-labels
- Log per-dimension attention entropy during validation

### `modal/siglip2_v2_data.py` — Data pipeline

- Class-balanced sampling option (inverse-frequency by quality bin)
- Multi-split DIQA loading (train + val with separate weights)

### New config files — `modal/configs/`

- 4 ablation configs (a1-a4)
- 2 dropout configs (0.1, 0.15)
- 1 sharpness-weighted config (conditional)

---

## Budget Summary

| Phase | Runs | Est. Cost | Cumulative |
|-------|------|-----------|------------|
| Phase 0 (agent analysis) | 0 | ~$0 (local) | $0 |
| Phase 1.1 (ablation) | 4 | ~$60 | $60 |
| Phase 1.2 (dropout) | 2 | ~$30 | $90 |
| Phase 1.3 (sharpness, conditional) | 1-2 | ~$30 | $120 |
| Phase 1.4 (teacher-student, conditional) | 2 | ~$40 | $160 |
| Phase 2 (pseudo-labels) | 3-4 | ~$60 | $220 |
| Phase 3 (OOD) | 2-3 | ~$45 + VLM API | $265+ |

Decision gates allow stopping early — if Phase 1 hits 0.92, skip Phase 2 entirely. Phase 1.3 and 1.4 are conditional on sharpness remaining the bottleneck.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Attention pooling overfits on 3,500 samples | Ablation A2 reveals this; fallback to mean pooling |
| PCGrad adds noise | Ablation A4 vs A3 comparison; skip if no benefit |
| Pseudo-labels too noisy | Step 2.1 validates on GT before training |
| Catastrophic forgetting with external data | Phase 3 is isolated; always measure DIQA regression |
| wSRCC plateaus at ~0.90 | Error analysis to find systematic failures; may be a data ceiling |
| A10 VRAM insufficient at 784 patches | Validate script already confirmed; fallback batch=2, grad_accum=8 |

---

## Verification

1. **Phase 1**: Compare per-run wSRCC with bootstrap CIs against v1.0 baseline (0.891)
2. **Phase 2**: Compare B1 (GT only) vs B2-B4 (pseudo-labels) to validate pipeline value
3. **Phase 3**: Measure DIQA test wSRCC before/after OOD data to confirm no regression
4. **All phases**: WandB logging for reproducibility, checkpoint saving for rollback
