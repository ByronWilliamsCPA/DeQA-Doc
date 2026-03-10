# Peer Review: Paper 9 — HyperIQA++ Training
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3 (`deepseek/deepseek-v3-0324`) was unavailable on OpenRouter during the review session. Consensus is based on 4 of 5 models. All other models completed successfully.

## Consensus Recommendation: Minor-to-Major Revision

The panel split evenly: **Minor Revision** (Gemini 3.1 Pro, Grok 4.1 Fast) vs **Major Revision** (GPT-5.2, Qwen 3.5+). The split reflects disagreement on whether the missing ablation study and hyperparameter omissions constitute minor gaps (addressable without new experiments) or major methodological shortcomings. All four reviewers agree the paper contains valuable findings but requires targeted revisions before publication.

## Aggregated Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 4 | **3.75** |
| Completeness | 2 | 3 | 3 | 3 | **2.75** |
| Clarity | 3 | 5 | 4 | 5 | **4.25** |
| Novelty | 3 | 4 | 3 | 3 | **3.25** |
| Reproducibility | 2 | 5 | 3 | 4 | **3.50** |
| **Overall** | | | | | **3.50** |

## Unanimous Findings

All four reviewers independently identified the same critical issues:

### 1. Missing Ablation Study (All 4 models)

The paper introduces four simultaneous modifications (spatial attention, multi-dimensional output, soft-label distribution heads, 1600x1600 resolution) but provides no controlled experiment isolating individual contributions. The 1.2-point gain over the competition baseline (0.856 vs 0.844) cannot be attributed to any specific extension.

**Actionable fix**: Add an ablation table with at minimum a leave-one-out design removing each extension individually, reporting both ID and OOD MainScore.

### 2. Unresolved MAE Calibration Anomaly (All 4 models)

MAE = 2.225 on a [1, 5] scale coexists with PLCC = 0.886, indicating a systematic scale offset. The paper correctly identifies this anomaly but only hypothesizes a root cause without experimental validation. Qwen 3.5+ called this a "calibration crisis."

**Actionable fix**: Apply isotonic regression or linear rescaling on a held-out calibration set and report both raw and calibrated MAE. Include a reliability diagram.

### 3. MainScore Inconsistency (All 4 models)

The abstract reports MainScore = 0.856 (line 15), but Table 5.3 (line 183) lists MainScore (ID) = 0.840. This 1.6-point discrepancy suggests different test splits or evaluation conditions that are not explained.

**Actionable fix**: Reconcile to a single canonical ID evaluation. Clarify whether 0.856 is on the full DIQA-5000 test set and 0.840 is on the OOD-benchmark's in-distribution subset (n=150).

### 4. Section 5.X Placeholder (All 4 models)

Section 5.X and the "see Section 5.X" cross-reference (line 145) are draft artifacts. Finalize numbering (presumably 5.5 or 5.4).

## Key Points of Disagreement

### Reproducibility: 2/5 vs 5/5

- **GPT-5.2 (2/5)**: Missing hyperparameters (learning rate, weight decay, batch size, epochs, seeds) and underspecified OOD dataset construction block replication.
- **Gemini 3.1 Pro (5/5)**: Detailed artifact paths (checkpoints, scripts, prediction files) provide exemplary transparency.
- **Resolution**: Both are partially right. The artifact pointers are excellent, but Section 4.3 explicitly states "Learning rate scheduling and weight decay values are not detailed here," which does block exact reproduction. A score of 3-4 is most appropriate.

### Clarity: 3/5 vs 5/5

- **GPT-5.2 (3/5)**: Metric inconsistencies and placeholder sections lower clarity.
- **Gemini & Grok (5/5)**: Writing quality is exceptional; inconsistencies are minor editorial issues, not structural clarity problems.
- **Resolution**: The writing quality is indeed high. The inconsistencies are fixable editorial issues that do not reflect poor organization or unclear communication.

## Consolidated Strengths

1. **Honest OOD generalization analysis**: The "off-the-shelf paradox" (fine-tuned 0.694 vs pretrained 0.723 on OOD) is a valuable negative result that most papers would omit. All four reviewers praised this transparency.

2. **Clear architectural positioning**: The CNN ceiling analysis (~0.86 MainScore) relative to ViT (~0.89) and MLLM (~0.93) tiers provides actionable guidance for practitioners choosing model architectures.

3. **Practical deployment value**: Inference speed comparisons (~100ms CNN vs ~3,000ms MLLM) and ensemble diversity rationale ground the work in production system design rather than pure benchmark chasing.

## Consolidated Weaknesses

1. **Confounded improvements**: Multiple simultaneous changes prevent attribution of gains. This is the most impactful weakness, cited by all four reviewers as the primary obstacle to the paper's conclusions.

2. **Incomplete training specification**: Section 4.3 explicitly omits critical hyperparameters. No training curves, no convergence analysis, no multi-seed variance reporting.

3. **Unvalidated calibration hypothesis**: The MAE anomaly is identified but not fixed. A simple linear rescaling experiment would resolve this in hours.

## Specific Revision Actions (Priority-Ordered)

### High Priority (Required for Acceptance)

1. **Add ablation table**: At minimum, test (a) base HyperIQA fine-tuned at 1600x1600, (b) +spatial attention, (c) +soft-label heads, (d) full HyperIQA++. Report both ID and OOD metrics.

2. **Fix MAE calibration**: Run isotonic regression or linear rescaling on held-out data. Report calibrated MAE alongside raw MAE.

3. **Reconcile MainScore values**: Ensure one canonical number throughout the paper. Explain the 0.856 vs 0.840 discrepancy.

4. **Report complete hyperparameters**: Learning rate, schedule, batch size, epochs, weight decay, warmup, and random seeds in Section 4.3.

### Medium Priority (Strengthens Paper)

5. **Add training convergence curves**: Show loss/MainScore over epochs for both ID and OOD validation to illustrate when catastrophic forgetting begins.

6. **Add bootstrap confidence intervals**: 95% CIs via bootstrap resampling (n=1,000) for all SRCC/PLCC values. The 3-point gap between HyperIQA++ and SigLIP2 may not be statistically significant.

7. **Soften "fundamental" ceiling language**: The CNN ceiling claim is plausible but not proven without broader architecture sweeps. Use "observed" rather than "fundamental."

8. **Add qualitative failure examples**: Include at least one figure showing OOD failure cases where CNN local receptive fields miss semantic quality cues.

### Low Priority (Polish)

9. **Fix citation year**: DBCNN is cited as (Zhang et al., 2018) in text but (Zhang et al., 2020) in references.

10. **Clarify parameter count claim**: Line 97-98 attributes the 28M-to-138M parameter increase to "larger spatial feature maps," but input resolution changes affect compute/activations, not parameter count. Clarify what architectural changes (spatial attention module, multi-head outputs) actually added parameters.

11. **Fix Section 5.X numbering**: Finalize to 5.5 or renumber sequentially.

12. **Precision in language**: "3 points below" (line 206) should be "3 percentage points below" or "0.030 lower."

13. **Training script path**: `image_detection/modal/train_hyperiqa_plus_plus.py` uses an `image_detection/` namespace that may confuse readers expecting an IQA-focused path.

14. **Reference formatting**: Inconsistent italicization of journal/conference names in the reference list.

## Individual Reviews

### GPT-5.2 (Neutral) — Major Revision

> Promising empirical results and a useful negative finding (ID/OOD trade-off), but several methodological ambiguities, missing ablations, and apparent metric/calibration inconsistencies prevent the current conclusions from being fully supported.

Scores: TS=3, C=2, Cl=3, N=3, R=2. Most critical reviewer. Unique observations: parameter count claim is technically incorrect (resolution does not change parameter count), over-strong causal language around catastrophic forgetting needs softening or evidence.

### Gemini 3.1 Pro (Neutral) — Minor Revision

> The paper presents a highly transparent, well-structured empirical analysis of document IQA fine-tuning, but requires minor revisions to address a metric calibration anomaly and missing architectural ablations before final acceptance.

Scores: TS=4, C=3, Cl=5, N=4, R=5. Most favorable reviewer. Praised exceptional writing quality and transparency. Unique observations: no qualitative visuals despite discussing spatial attention and failure modes; missing inline figure references despite mentioning a figure generation script.

### Qwen 3.5+ (Neutral) — Major Revision

> The paper provides valuable empirical findings about CNN-based document IQA, particularly the honest OOD generalization analysis. However, the missing ablation studies, incomplete hyperparameter specification, and unresolved MAE calibration issue prevent the work from being fully actionable or reproducible.

Scores: TS=4, C=3, Cl=4, N=3, R=3. Unique observations: suggested 2^4 factorial ablation design; recommended training convergence curves to pinpoint when forgetting occurs; noted training script path namespace mismatch.

### Grok 4.1 Fast (Neutral) — Minor Revision

> Writing is precise, well-organized with logical flow from baseline to extensions/results/discussion; tables are effective and self-explanatory. Strongest novelty in positioning CNN ceiling vs. ViT/MLLM.

Scores: TS=4, C=3, Cl=5, N=3, R=4. Unique observations: detailed artifact links partially compensate for missing hyperparameters; incremental contributions are well-framed within the document IQA context.

### DeepSeek V3 — Unavailable

Model ID `deepseek/deepseek-v3-0324` was not recognized by OpenRouter. Review not completed.
