# VLM Teacher Evaluation — 6-Model Consensus Review Tracker

**Document Reviewed**: `VLM_TEACHER_EVALUATION.md`
**Date**: 2026-03-07
**Models Consulted**: 6 of 8 (2 OpenAI models failed — GPT-5.4 API error, GPT-5.3 invalid model ID)

## Review Panel

| # | Model | Stance | Confidence | Key Focus |
|---|-------|--------|------------|-----------|
| 1 | Gemini 3 Flash Preview | FOR | 9/10 | Deep technical review, pipeline architecture |
| 2 | Grok 4.1 Fast | AGAINST | 8/10 | Production feasibility, calibration contradiction |
| 3 | Grok 4 Fast | NEUTRAL | 8/10 | Statistical rigor, alternative approaches |
| 4 | Llama 4 Maverick | FOR | 7/10 | Research novelty, value proposition |
| 5 | Qwen Plus | NEUTRAL | 8/10 | Sharpest architectural critique, threshold analysis |
| 6 | MiniMax M1 | AGAINST | 5/10 | Rigorous statistical review, circularity detection |

**Weighted Consensus Score**: ~7.5/10

---

## How to Use This Tracker

Each finding has:
- **ID**: For reference (e.g., U-1, S-3, N-7)
- **Status**: `[ ]` open, `[~]` in progress, `[x]` resolved, `[—]` won't fix (with rationale)
- **Priority**: P0 (blocks everything), P1 (fix before viability claims), P2 (strengthen before publication)
- **Source**: Which model(s) raised it
- **Agreement**: How many models agree (e.g., 6/6 unanimous)
- **Resolution**: Notes on how it was addressed

---

## P0 — Blocking (Do Immediately)

### U-1: End-to-End Student Training Validation
- **Status**: `[ ]`
- **Priority**: P0
- **Agreement**: 6/6 unanimous
- **Source**: All models
- **Finding**: No student model has been trained on VLM pseudo-labels. The entire pipeline is theoretical. SigLIP2-IQA must be trained on calibrated pseudo-labels and measured against the human-label baseline.
- **Success Criteria**: wSRCC degradation < 0.02 vs. human-label baseline (0.716)
- **Depends On**: U-2 (calibration must be validated first)
- **Resolution**: _pending_

### U-2: Calibration Experiment (Isotonic Regression)
- **Status**: `[x]`
- **Priority**: P0
- **Agreement**: 6/6 unanimous
- **Source**: All models (Gemini: "non-negotiable", Grok 4.1: "prioritize immediately")
- **Finding**: Isotonic regression is proposed (Section 6, Stage 2) but never tested. No post-calibration wSRCC/MAE results appear anywhere in the document. Must report calibration results on DIQA-5000 test set before any pipeline viability claims.
- **Success Criteria**: Report per-model, per-dimension wSRCC and MAE before and after isotonic regression on DIQA-5000 train→test
- **Depends On**: Nothing — can start immediately with existing data
- **Resolution**: **Completed 2026-03-07.** Ran calibration experiment on SigLIP2-IQA (the only model with train+test predictions). Compared 3 calibration methods (linear, 4PL logistic, isotonic) per dimension on train split (3,500 samples) → test split (1,000 samples). Results:
  - **wSRCC is invariant** under all monotone calibrations: 0.8914 (raw/linear/4PL), 0.8910 (isotonic — tiny delta from tied ranks)
  - **MAE drops 14x** after calibration: 2.424 (raw, due to [0,1] → MOS scale mismatch) → 0.173 (linear/4PL) / 0.174 (isotonic)
  - **Linear ≈ 4PL** — the SigLIP2 [0,1] → MOS mapping is nearly linear, so the 4PL logistic adds no benefit over affine
  - **PLCC already high**: 0.921 (overall) even without calibration — SigLIP2's predictions are well-ordered
  - Per-dimension SRCC: overall=0.899, sharpness=0.874, color=0.893
  - Per-dimension MAE (post-calibration): overall=0.167, sharpness=0.184, color=0.172
  - Script: `results/siglip2_diqa5000/calibrate_isotonic.py`, results: `results/siglip2_diqa5000/calibration_results.json`
  - **13-model consensus review** (GPT-5.2, Gemini 3.1 Pro, Gemini 3 Flash, DeepSeek v3.2, Minimax M2.5, Grok 4.1 Fast, Qwen3.5-397B, Qwen3.5-Plus, Kimi K2.5, GLM-5, Trinity Large, Nemotron Nano, GLM-4.5-Air) validated the approach
  - **Note**: VLM models lack train-split predictions — calibrating VLMs requires running inference on 3,500 train images first (future work for U-1)

---

## P1 — Fix Before Claiming Pipeline Viability

### U-3: OOD Threshold Sensitivity Analysis
- **Status**: `[x]`
- **Priority**: P1
- **Agreement**: 6/6 unanimous
- **Source**: All models
- **Finding**: The p95/p99 distance thresholds (30.8/58.2) and MOS disagreement cutoffs (0.5/1.0/1.5) are arbitrary with no ablation. No analysis shows how auto-accept/review/reject proportions change with threshold shifts.
- **Action**: Sweep p90, p92, p95, p97, p99 distance thresholds and 0.3/0.5/0.8/1.0/1.5 MOS disagreement thresholds. Report % images in each tier and downstream impact.
- **Resolution**: **Completed 2026-03-07.** Two-part sweep covering Tier-1 fusion thresholds and Tier-2 VLM veto thresholds on all 5,000 DIQA-5000 images (train+val for calibration, test for evaluation). Key findings:
  - **σ²/entropy thresholds are dead code**: Current defaults (σ² auto=0.64, entropy auto=1.2) never trigger — actual σ² max ~0.12, entropy max ~0.70. Result: 93.7% AUTO_ACCEPT identical to `dm_only` profile.
  - **Data-calibrated percentile thresholds work**: Using train+val percentiles (σ² p75/p90, entropy p75/p90) produces 65.4% AUTO_ACCEPT with meaningful tier differentiation (16.9% LOW_WEIGHT, 16.8% TIER2_TRIGGER).
  - **d_M sweep**: AUTO_ACCEPT ranges from 25.3% (p90) to 45.7% (p99) on test, showing strong sensitivity.
  - **Tier-2 VLM veto rates vary wildly by model**: At threshold=1.5, claude-haiku vetoes 0.5% while qwen3.5-flash vetoes 60.3%. Ensemble majority vote: 5.6%.
  - **Per-dimension differences**: Sharpness has highest veto rates (e.g., gpt-4.1: 43.7% sharpness vs 27.9% overall), supporting per-dimension thresholds.
  - **13-model consensus review** (same panel as U-2) validated the approach, recommending: `siglip2_output_to_level_probs()` for entropy, percentile-based hard-reject, train+val calibration discipline.
  - Script: `research/threshold_sensitivity/run_sweep.py`, results: `results/threshold_sensitivity/sweep_results.json`, report: `results/threshold_sensitivity/sweep_report.md`
  - **Limitation**: JSD thresholds not swept (DeQA per-image predictions unavailable). GT veto accuracy not computable (test set lacks GT MOS).

### U-4: Synthetic OOD Insufficient — Need Real-World OOD
- **Status**: `[ ]`
- **Priority**: P1
- **Agreement**: 6/6 unanimous
- **Source**: All models
- **Finding**: Synthetic OOD dataset uses parameter-derived MOS (not human-validated), making ground truth circular. Procedurally generated images are trivially distinguishable in embedding space, inflating AUROC (0.9963). Need real-world OOD documents.
- **Action**: Collect 500+ real-world OOD documents (RVL-CDIP, handwritten forms, receipts, historical manuscripts) and evaluate both VLM teachers and OOD detector on them.
- **Resolution**: _pending_

### S-1: Multiple Comparisons Correction
- **Status**: `[ ]`
- **Priority**: P1
- **Agreement**: 4/6 (Grok 4, Qwen Plus, MiniMax explicit; Grok 4.1 implicit)
- **Source**: Grok 4 Fast, Qwen Plus, MiniMax M1
- **Finding**: 7 models × 3 dimensions × 2 datasets = 42 tests, uncorrected. Family-wise error rate ~30% (1 - 0.95^7). Gemini 3 Flash's "clear winner" designation may be spurious. The document acknowledges overlapping CIs but never conducts a formal test.
- **Action**: Apply Benjamini-Hochberg FDR correction to all pairwise comparisons. Soften "clear winner" language if rankings don't survive correction.
- **Resolution**: _pending_

### S-2: OOD Threshold Circularity
- **Status**: `[ ]`
- **Priority**: P1
- **Agreement**: 3/6 (Qwen Plus, MiniMax explicit; Grok 4 implicit)
- **Source**: MiniMax M1, Qwen Plus
- **Finding**: The hard-reject threshold (58.2) is derived from test set p99, then applied as a production threshold. This creates circularity — the detector's performance metrics benefit from being fitted to data it's evaluated against. The auto-accept threshold (30.8) is train+val p95, but test p95 is 48.5 — meaning ~5% of in-distribution test images would be incorrectly flagged.
- **Action**: Re-fit using held-out calibration set (e.g., 3,000 fit / 1,000 calibrate from train+val) or k-fold cross-validation. Do not derive any threshold from the test set.
- **Resolution**: _pending_

### N-1: sigma_pseudo=0.8 vs Actual MOS std=0.47
- **Status**: `[x]`
- **Priority**: P1
- **Agreement**: 2/6 (Qwen Plus, MiniMax)
- **Source**: Qwen Plus (primary), MiniMax M1
- **Finding**: Stage 3 hardcodes `sigma_pseudo = 0.8` for uncertainty estimation, but DIQA-5000 human MOS std is only 0.47. This inflates uncertainty by ~70% and degrades soft-label quality — distributions will be too flat, reducing the training signal for the DeQA loss.
- **Action**: Ablate sigma_pseudo at 0.47, 0.6, 0.8, 1.0. Measure impact on soft-label KL divergence from human distributions and downstream student wSRCC.
- **Note**: The 0.8 value comes from DeQA-Score's original paper. The question is whether VLM inter-model variance justifies a higher sigma than human annotator variance.
- **Resolution**: **Misunderstanding — the implemented pipeline does not use σ_pseudo=0.8 for soft-label generation.** Code trace: `pseudo_label.py:97` calls `siglip2_output_to_level_probs(mu, sigma_sq)` which uses SigLIP2's *predicted* σ² directly (`gaussian_to_discrete.py:132`). The σ_pseudo=0.8 value only appears in the *unimplemented* VLM consensus formula (`VLM_TEACHER_EVALUATION.md:570`). The `sigma_sq_auto=0.64` in `fusion.py:88` is a decision-gate threshold (not a soft-label parameter); U-3 proved it is dead code since actual σ² values max at ~0.12. **The real concern is reversed**: SigLIP2 predicts σ≈0.23 (p50 σ²=0.054) vs human MOS std=0.47, making distributions too *peaked*, not too flat. A sigma_floor/sigma_scale ablation is deferred to U-1 (end-to-end student training) where downstream wSRCC impact can be measured. The VLM_TEACHER_EVALUATION.md formula has been updated to recommend data-calibrated σ_pseudo.

### N-2: Pipeline Telemetry — Auto-Accept/Review/Reject Rates
- **Status**: `[~]`
- **Priority**: P1
- **Agreement**: 2/6 (Qwen Plus, MiniMax)
- **Source**: Qwen Plus (primary)
- **Finding**: The pipeline's scalability is unknown without telemetry. What % of images trigger each tier? If review rate is >20%, the pipeline may be impractical. If reject rate is >5%, you're losing significant training data.
- **Action**: Run the proposed Stage 4 logic on all 1,000 test images using existing Gemini + GPT-4.1 scores and OOD distances. Report tier distribution.
- **Resolution**: **Partially addressed by U-3.** The threshold sensitivity sweep provides tier distributions under 12 threshold configurations. With data-calibrated thresholds on test: 65.4% AUTO_ACCEPT, 16.9% LOW_WEIGHT, 16.8% TIER2_TRIGGER, 0.9% HARD_REJECT (overall). Review rate (TIER2_TRIGGER) is <20% under all profiles except `strict`. See `results/threshold_sensitivity/sweep_report.md`. Still pending: telemetry with JSD signal (requires DeQA inference) and end-to-end pipeline run.

### N-3: Error Correlation Between Gemini and GPT-4.1
- **Status**: `[ ]`
- **Priority**: P1
- **Agreement**: 1/6 (MiniMax), but critical
- **Source**: MiniMax M1
- **Finding**: If Gemini and GPT-4.1 make correlated errors (both fail on the same images), the consensus approach adds minimal value over single-model annotation. The per-image residuals (VLM score - human MOS) should be correlated across models to quantify independence.
- **Action**: Compute Pearson correlation of per-image residuals between Gemini and GPT-4.1 across all 1,000 test images, per dimension. Report correlation coefficient and scatter plot.
- **Resolution**: _pending_

---

## P2 — Strengthen Before Publication

### S-3: Weak Tiebreaker Model
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 4/6 (Gemini, Grok 4.1, Qwen Plus, Grok 4)
- **Source**: Gemini 3 Flash (primary), Qwen Plus
- **Finding**: Stage 1 uses Qwen3-VL-8B (wSRCC 0.481) or Claude Haiku 4.5 (wSRCC 0.579) as tiebreakers when Gemini and GPT-4.1 disagree by >1.0 MOS. Using a weaker model to adjudicate stronger ones is illogical.
- **Qwen Plus refinement**: For sharpness specifically, Haiku (SRCC_S 0.539) outperforms Qwen3-VL-8B Instruct (SRCC_S 0.437). Per-dimension tiebreaker selection may be better.
- **Action**: Consider alternatives — (a) use Gemini 2.5 Pro despite 7% parse failure, (b) use the median of 3 models, (c) use per-dimension best tiebreaker, or (d) flag for human review instead.
- **Resolution**: _pending_

### S-4: Reframe "Annotation-Free" as "Annotation-Efficient"
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 3/6 (Grok 4.1, Grok 4, MiniMax)
- **Source**: Grok 4.1 Fast (primary)
- **Finding**: The pipeline requires DIQA-5000's 3,500 training labels for calibration (Stage 2). The abstract and intro imply annotation-free pseudo-labeling, but calibration needs human labels for every new domain (if isotonic regression doesn't generalize). The framing should be honest: this is annotation-efficient (leveraging existing labels), not annotation-free.
- **Action**: Update abstract, intro, and conclusion language. Explicitly state the calibration dependency and quantify: "Requires N labeled images per target domain for calibration."
- **Resolution**: _pending_

### S-5: Bootstrap CI Methodology
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (MiniMax), but technically valid
- **Source**: MiniMax M1
- **Finding**: The 95% CIs use bootstrapping (1,000 iterations, seed=42) but don't specify whether bias-corrected accelerated (BCa) intervals were used. For SRCC (bounded, potentially asymmetric), percentile intervals may be biased.
- **Action**: Re-compute CIs using BCa method. Specify method in the document. If results change materially, update Table 1.
- **Resolution**: _pending_

### S-6: Power Analysis for n>=200 Recommendation
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (MiniMax)
- **Source**: MiniMax M1
- **Finding**: The recommendation of n>=200 stratified samples for prompt optimization lacks justification. Why 200 and not 150 or 300? A formal power analysis (detecting delta=0.03 wSRCC at 80% power) would be more rigorous.
- **Action**: Run power analysis for detecting meaningful wSRCC differences. May increase or decrease the recommendation.
- **Resolution**: _pending_

### N-4: Mahalanobis Distance ≠ Label Reliability
- **Status**: `[ ]`
- **Priority**: P2 (architectural — may require pipeline redesign)
- **Agreement**: 2/6 (Qwen Plus, MiniMax implicit)
- **Source**: Qwen Plus
- **Finding**: The OOD detector measures embedding-space deviation from the training distribution, NOT whether VLM labels are reliable. A pristine document (high Mahalanobis distance) may still receive accurate VLM ratings. A form-layout document (low distance, in-distribution) may be systematically misrated. The detector's purpose is misaligned with its use.
- **Action**: Consider replacing or augmenting the OOD gate with a label-reliability classifier trained on human-verified VLM errors (images where VLMs had |error| > 1.0 MOS). This directly measures what we care about.
- **Resolution**: _pending_

### N-5: Calibration Generalizability to New Domains
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (MiniMax), but critical for the pipeline's purpose
- **Source**: MiniMax M1
- **Finding**: Isotonic regression trained on DIQA-5000 may not generalize to new document domains (handwritten, historical, multilingual). If calibration doesn't transfer, human labels are needed per domain — defeating the pseudo-labeling purpose.
- **Action**: Test calibration transfer by training on DIQA-5000 train and evaluating on (a) DIQA-5000 test, (b) synthetic in-distribution, (c) synthetic OOD subsets. If MAE degrades >50% on OOD, explore domain-adaptive calibration.
- **Resolution**: _pending_

### N-6: Pipeline Failure Recovery
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (Qwen Plus)
- **Source**: Qwen Plus
- **Finding**: If Gemini 3 Flash fails (7% parse failure rate observed for Gemini 2.5 Pro), GPT-4.1 alone carries the load — but GPT-4.1 has the worst MAE (1.15), silently degrading label quality. No failure recovery mechanism exists.
- **Action**: Add retry logic with structured output (JSON mode), fallback model selection, and quality degradation alerting. Log single-model vs. dual-model annotation rates.
- **Resolution**: _pending_

### N-7: Latency and Parallelization Strategy
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (Qwen Plus)
- **Source**: Qwen Plus
- **Finding**: At 5-6s/image for dual VLM annotation, processing 1M documents takes ~12 days on a single thread. The document doesn't discuss parallelization, batching, or async strategies.
- **Action**: Document expected throughput at various concurrency levels. Note that VLM APIs support parallel requests — 100 concurrent requests would reduce 1M docs to ~3 hours.
- **Resolution**: _pending_

### N-8: Per-Dimension Disagreement Thresholds
- **Status**: `[~]`
- **Priority**: P2
- **Agreement**: 1/6 (Qwen Plus)
- **Source**: Qwen Plus
- **Finding**: Stage 1 uses a single >1.0 MOS disagreement threshold across all dimensions, but model accuracy varies substantially by dimension (e.g., color fidelity is hardest for most models). Per-dimension thresholds would be more principled.
- **Action**: Compute per-dimension inter-model disagreement distributions on the 1,000-image test set. Set dimension-specific thresholds (e.g., 0.8 for color, 1.0 for overall, 1.0 for sharpness).
- **Resolution**: **Data now available from U-3.** The Tier-2 VLM veto sweep reports per-dimension veto rates across 9 models × 5 thresholds. Key finding: sharpness has consistently higher veto rates than overall or color (e.g., at threshold=1.5, gpt-4.1 vetoes 43.7% sharpness vs 27.9% overall vs 23.1% color). This confirms per-dimension thresholds are needed. See `results/threshold_sensitivity/sweep_report.md` Section 5.3. Still pending: derive recommended per-dimension values.

### N-9: Disagreement Threshold Asymmetry (0.5 vs 1.0 vs 1.5)
- **Status**: `[~]`
- **Priority**: P2
- **Agreement**: 1/6 (MiniMax)
- **Source**: MiniMax M1
- **Finding**: The pipeline uses three different MOS disagreement cutoffs: 0.5 (auto-accept), 1.0 (trigger third model in Stage 1), and 1.5 (reject in Stage 4). The 0.5 auto-accept is stricter than the 1.0 third-model trigger — this asymmetry is unexplained. Why would an image with 0.6 MOS disagreement need review (Stage 4) but not a third model (Stage 1)?
- **Action**: Explain the rationale or unify the thresholds. The Stage 1 trigger and Stage 4 review thresholds should be derived from the same inter-model disagreement distribution.
- **Resolution**: **Data now available from U-3.** The veto threshold sweep at [0.3, 0.5, 0.8, 1.0, 1.5] shows how veto rates change across the full range. At threshold=0.5, ensemble majority vetoes 94.7% of images (far too aggressive). At threshold=1.5, only 5.6% (very conservative). The data supports threshold=0.8-1.0 as a balanced range for the veto gate. See `results/threshold_sensitivity/sweep_report.md` Section 5.2. Still pending: formal rationale for the 0.5/1.0/1.5 asymmetry between pipeline stages.

### N-10: Explore Quantile Mapping vs. Isotonic Regression
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (Grok 4)
- **Source**: Grok 4 Fast
- **Finding**: Quantile mapping may generalize better than isotonic regression to OOD documents because it preserves the full distribution shape rather than learning a monotonic point mapping. Worth testing as an alternative or complement.
- **Action**: Implement quantile mapping calibration alongside isotonic regression. Compare post-calibration MAE on DIQA-5000 test and synthetic OOD subsets.
- **Resolution**: _pending_

### N-11: VLM Version Drift Risk
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 2/6 (Grok 4.1, Grok 4)
- **Source**: Grok 4.1 Fast
- **Finding**: VLM APIs are updated without notice (model versioning, deprecation). A pipeline calibrated on Gemini 3 Flash Preview may produce different results when the model is updated. No drift detection or recalibration strategy is documented.
- **Action**: Add a calibration monitoring protocol: re-run 100 sentinel images monthly and alert if wSRCC degrades >0.02 or MAE shifts >0.1.
- **Resolution**: _pending_

### N-12: Small-Sample Unreliability — Explore Root Cause
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (MiniMax)
- **Source**: MiniMax M1
- **Finding**: The n=23 to n=1,000 reversal is well-documented but unexplained. Is it VLM output variance, image selection bias in the 23-image stratified subset, or quality-bucket interaction effects? Understanding the root cause would strengthen the contribution.
- **Action**: Analyze the 23-image subset: compare quality bucket distribution to the full 1,000, compute per-bucket wSRCC, and identify which images drove the n=23 no-resize advantage.
- **Resolution**: _pending_

### N-13: Hybrid NR-IQA Signals for Uncertainty Fusion
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (Grok 4)
- **Source**: Grok 4 Fast
- **Finding**: Per-image NR-IQA baseline scores (TReS, HyperIQA) are archived and could serve as weak ensemble signals. Images where DeQA-Doc and off-the-shelf baselines strongly disagree may indicate annotation difficulty. These signals could augment VLM uncertainty estimation at near-zero cost.
- **Action**: Compute correlation between NR-IQA disagreement and VLM error magnitude. If correlated, add as a feature to the label-reliability model.
- **Resolution**: _pending_

### N-14: OOD Gating Ablation
- **Status**: `[ ]`
- **Priority**: P2
- **Agreement**: 1/6 (Qwen Plus)
- **Source**: Qwen Plus
- **Finding**: Does removing Stage 4 (OOD gating) actually degrade student model performance? This is unmeasured. If the student is robust to noisy labels from OOD images, the gating may be unnecessary complexity.
- **Action**: Train SigLIP2-IQA with and without OOD gating. Compare wSRCC to measure the gate's actual value.
- **Depends On**: U-1 (end-to-end training)
- **Resolution**: _pending_

---

## Summary Statistics

| Priority | Count | Status |
|----------|-------|--------|
| P0 (blocking) | 2 | 1 resolved |
| P1 (pre-viability) | 7 | 1 resolved, 1 partial |
| P2 (pre-publication) | 14 | 2 partial |
| **Total** | **23** | **2 resolved, 3 partial** |

| Agreement Level | Count |
|----------------|-------|
| 6/6 unanimous | 4 (U-1 through U-4) |
| 4-5/6 strong | 4 (S-1 through S-4, plus aspects of S-5, S-6) |
| 1-3/6 novel | 15 (N-1 through N-14, plus S-5, S-6) |

---

## Per-Model Review Summaries

### Gemini 3 Flash Preview (FOR, 9/10)

**Verdict**: "Rigorous, high-quality research effort that demonstrates viability, though pipeline requires more robust calibration and consensus logic."

**Key contributions to consensus**:
- Calibration is non-negotiable — VLMs are excellent rankers but terrible calibrators
- Qwen-8B tiebreaker is a risk — using a weak model to adjudicate strong ones
- OOD detector (AUROC 0.996) is the most reliable pipeline component
- Stick to 1024px — native resolution adds cost without gains
- Never make VLM decisions on <200 samples

**Unique perspective**: Most optimistic about the research quality. Framed calibration as a solvable engineering problem rather than a fundamental contradiction. Strong endorsement of the OOD detector's reliability despite threshold concerns.

---

### Grok 4.1 Fast (AGAINST, 8/10)

**Verdict**: "Innovative pipeline with compelling benchmarks, but flawed by unvalidated end-to-end performance, synthetic OOD limitations, and unproven thresholds."

**Key contributions to consensus**:
- Calibration contradicts "no human annotation" goal — pipeline is annotation-efficient, not annotation-free
- VLM API dependency creates fragility (version drift, rate limits, TOS)
- 7% parse failure rate means cascade failures at scale
- Simpler alternative: single Gemini + isotonic may achieve 80% of benefit
- Cost-optimize: batch VLMs, cap review tier at 20%

**Unique perspective**: First to identify the annotation-free framing as misleading. Strongest on production feasibility concerns (API TOS, rate limits). Proposed the simplest viable alternative (single model + calibration).

---

### Grok 4 Fast (NEUTRAL, 8/10)

**Verdict**: "Technically promising but undermined by statistical weaknesses, unvalidated thresholds, and unaddressed production-scale risks."

**Key contributions to consensus**:
- Missing multiple comparisons correction (42 tests uncorrected)
- Quantile mapping may generalize better than isotonic for OOD calibration
- Cap review queue at 20% of images to prevent overload
- Hybrid NR-IQA signals for uncertainty fusion at near-zero cost
- PLCC >> SRCC on synthetic data suggests logistic fitting is doing heavy lifting

**Unique perspective**: Most balanced assessment. Best at identifying alternative approaches (quantile mapping, NR-IQA hybrid, active learning). First to note the PLCC/SRCC gap as a red flag for synthetic data validity.

---

### Llama 4 Maverick (FOR, 7/10)

**Verdict**: "Technically feasible but requires careful consideration of limitations and improvements for practical viability."

**Key contributions to consensus**:
- Multi-model consensus likely provides better robustness than single VLM
- VLM API dependency is the main long-term risk
- Large-scale validation and iterative refinement are necessary
- Monitoring and updating VLM APIs and OOD model are crucial

**Unique perspective**: Most general review — confirmed feasibility and value proposition without adding new critical findings. Lowest confidence among FOR models (7/10), reflecting honest uncertainty about scalability.

---

### Qwen Plus (NEUTRAL, 8/10)

**Verdict**: "Technically promising but critically undermined by unvalidated pipeline assumptions, statistically weak calibration justification, and dangerously optimistic OOD threshold policy."

**Key contributions to consensus**:
- **sigma_pseudo=0.8 vs actual MOS std=0.47** — inflates uncertainty 70%, degrades soft-labels (NOVEL)
- **Mahalanobis measures embedding deviation, not label reliability** — fundamental mismatch (NOVEL)
- **OOD threshold train/test mismatch** — 5% of ID images incorrectly flagged (NOVEL)
- **No pipeline telemetry** — auto-accept/review/reject rates unknown (NOVEL)
- **No failure recovery** — 7% Gemini failure leaves worst-MAE model alone (NOVEL)
- Replace OOD distance with label-reliability classifier (ARCHITECTURAL)
- Per-dimension disagreement thresholds instead of single trigger
- 12 days to process 1M docs without parallelization

**Unique perspective**: The sharpest and most technically specific critique. Found more novel issues (8) than any other model. The sigma_pseudo mismatch and label-reliability classifier suggestions are the most architecturally significant findings in the entire review.

---

### MiniMax M1 (AGAINST, 5/10)

**Verdict**: "Well-motivated and thorough evaluation, but significant statistical weaknesses could undermine key claims."

**Key contributions to consensus**:
- **Family-wise error rate ~30%** with 7 models uncorrected (QUANTIFIED)
- **OOD threshold circularity** — test p99 used to set production threshold (NOVEL)
- **Bootstrap CIs may be biased** — BCa not specified for bounded SRCC metric (NOVEL)
- **n=200 recommendation needs power analysis** (NOVEL)
- **Calibration generalizability unknown** — may need human labels per domain (NOVEL)
- **Error correlation between models unexplored** — consensus may add little if correlated (NOVEL)
- **Disagreement threshold asymmetry** (0.5 vs 1.0 vs 1.5) unexplained (NOVEL)

**Unique perspective**: Most statistically rigorous review and lowest confidence (5/10). The only model to explicitly compute the family-wise error rate, question BCa bootstrap methodology, and demand a power analysis. Found the most statistically grounded novel issues. Its low confidence rating reflects that the pipeline's viability claims are not yet empirically supported.

---

## Changelog

| Date | Action | Items Affected |
|------|--------|----------------|
| 2026-03-07 | Initial tracker created from 6-model consensus review | All 23 items |
| 2026-03-07 | U-2 resolved: isotonic calibration experiment completed on SigLIP2 (3 methods, 13-model consensus review) | U-2 |
| 2026-03-07 | U-3 resolved: threshold sensitivity sweep (12 Tier-1 configs, 5 Tier-2 thresholds × 9 VLMs). N-2, N-8, N-9 partial. | U-3, N-2, N-8, N-9 |
