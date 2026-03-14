# Peer Review: Paper 7 --- Pseudo-Labeling Pipeline
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3-0324 was unavailable on OpenRouter at review time. Four models provided full reviews. DeepSeek V3.2 was attempted as a substitute but could not be routed the paper content through the available tooling. The synthesis below reflects the 4-model consensus from GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, and Grok 4.1 Fast.

## Consensus Recommendation: Major Revision

All four reviewers recommend **Major Revision**. The pipeline design is praised as rigorous and well-motivated, but the absence of end-to-end validation prevents the central claim from being empirically demonstrated.

---

## Scores Summary

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | Mean |
|-----------|---------|----------------|------------|---------------|------|
| Technical Soundness | 3 | 4 | 4 | 5 | **4.0** |
| Completeness | 2 | 3 | 3 | 4 | **3.0** |
| Clarity | 4 | 5 | 4 | 5 | **4.5** |
| Novelty | 3 | 4 | 4 | 4 | **3.75** |
| Reproducibility | 3 | 4 | 4 | 5 | **4.0** |
| **Weighted Mean** | | | | | **3.85** |

---

## Unanimous Findings (4/4 reviewers agree)

### 1. No end-to-end validation cycle has been executed

The single most critical gap. The paper proposes an iterative pseudo-labeling pipeline whose central claim is that it safely expands domain coverage, but no complete cycle (pseudo-label -> retrain -> re-evaluate) has been run. All reviewers flag this as the blocking issue that prevents acceptance.

> "The paper describes a system whose primary purpose is iterative retraining, but stops before demonstrating a single retraining cycle." --- Gemini 3.1 Pro

> "The main claim (iterative pseudo-labeling expands domain coverage safely) is not yet empirically demonstrated." --- GPT-5.2

### 2. Synthetic-only OOD evaluation

All OOD detection and cross-domain results use programmatically generated documents. Real-world document diversity (handwritten forms, historical manuscripts, receipts) may produce different detection and labeling characteristics. The paper itself acknowledges this risk, and all reviewers emphasize that synthetic AUROC likely overestimates real-world performance.

### 3. Calibration demonstrated on student model, not VLM teachers

The 14x MAE reduction is demonstrated on SigLIP2 predictions, not on VLM teacher outputs --- the actual use case for the pipeline. VLM calibration is described as "planned" but not executed. Whether isotonic calibration generalizes across document types remains an open question.

### 4. Exceptional writing quality and transparency

All reviewers praise the paper's clarity, organization, and honest reporting of limitations. The explicit failure mode analysis, cost breakdown, and candid acknowledgment of missing validation are cited as exemplary.

---

## Majority Findings (3/4 reviewers agree)

### 5. "14x MAE reduction" headline is misleading (3/4)

Three reviewers note that the 14x reduction is primarily attributable to a scale mismatch (SigLIP2 outputs on [0,1] vs. MOS on [1,5]), not a sophisticated calibration challenge. The paper should reframe this claim to foreground the scale mismatch cause.

### 6. Missing pipeline ablation study (3/4)

No systematic ablation isolates the contribution of each pipeline stage (OOD gate on/off, single vs. dual teacher, weighting schemes). Without this, the relative importance of each component is unknown.

### 7. Section 6.3 formatting bug (3/4)

The numbered lists for "Medium-term directions" and "Longer-term research" use repeated "1." instead of incrementing numbers (1, 2, 3, 4).

---

## Individual Reviewer Highlights

### GPT-5.2 (Strictest Reviewer)

**Scores**: TS=3, C=2, CL=4, N=3, R=3 | **Recommendation**: Major Revision

Key unique points:
- The "14x MAE reduction" headline risks being misread as a sophisticated calibration win; should foreground the scale mismatch cause
- Clarify "MainScore" definition when mixing wSRCC and MainScore across tables to avoid metric confusion
- The gating description mentions Tier 2 cross-validation with Qwen3-VL-8B but the teacher/tiebreaker section emphasizes other models; tighten the exact decision policy
- Missing: exact prompts/templates, parsing rules, model versioning, synthetic OOD generation recipe, student retraining hyperparameters

### Gemini 3.1 Pro (Most Detailed)

**Scores**: TS=4, C=3, CL=5, N=4, R=4 | **Recommendation**: Major Revision

Key unique points:
- The paper currently acts more as "a robust architecture proposal than a fully realized iterative system"
- Calibration distribution shift is theoretically risky given documented VLM capriciousness
- Suggests a small validation experiment: take human-labeled OOD documents, apply ID-fitted calibration curve to VLM scores, verify MAE stays low
- References Figures 1, 2, 3 which are generated via script but not embedded or described in sufficient depth

### Qwen 3.5+ (Balanced)

**Scores**: TS=4, C=3, CL=4, N=4, R=4 | **Recommendation**: Major Revision

Key unique points:
- The sigma-squared threshold (0.64) and entropy threshold (1.2) are effectively dead code --- never trigger on real DIQA-5000 data
- Suggests data-calibrated thresholds should be validated as the default configuration
- Notes the integration of OOD gating + VLM pseudo-labeling + learned calibration for document IQA domain expansion is genuinely novel despite individual components being established
- Recommends cross-dataset generalization experiments (Tobacco800, RVL-CDIP, CORD)

### Grok 4.1 Fast (Most Generous)

**Scores**: TS=5, C=4, CL=5, N=4, R=5 | **Recommendation**: Major Revision (implied)

Key unique points:
- Rates reproducibility 5/5 due to exceptional artifact documentation (embeddings, predictions, scripts, costs)
- Notes inconsistent model naming: "Gemini 3 Flash Preview" vs. "Gemini 3 Flash" --- should standardize
- Table header ambiguities in Appendix B: "d_M OOD" should be "d_M Tier2"; "sigma_sq Auto" unclear
- Suggests weighted ensemble ablation for VLM consensus, particularly per-category weighting based on VLM strengths

---

## Consolidated Strengths

1. **Rigorous component-level validation**: OOD detector AUROC = 0.9963, calibration 14x MAE reduction, VLM teacher wSRCC = 0.708 approaching supervised baseline of 0.716. Each stage is independently validated with appropriate statistics (bootstrap CIs, p-values).

2. **Exceptional transparency and honest limitation reporting**: The paper explicitly identifies where VLMs fail (binarized, extreme DPI, pristine), acknowledges missing end-to-end validation, and maps hard boundaries of the approach. Multiple reviewers call this "exemplary."

3. **Compelling practical motivation**: The 3,000x cost reduction vs. human annotation ($15 vs. $50,000+ per 5,000 images), downstream OCR-IQA correlation validation (SRCC up to -0.683), and clear domain expansion rationale make the pipeline immediately actionable.

4. **Strong downstream validation**: The OCR-IQA correlation study confirms that quality scores predict real-world document processing accuracy, proving the pipeline optimizes a meaningful proxy rather than a disconnected benchmark metric.

---

## Consolidated Weaknesses

1. **No end-to-end validation** (unanimous, blocking): The iterative expansion cycle --- the paper's central contribution --- has never been executed. Without demonstrating that SigLIP2 maintains SRCC > 0.90 on DIQA-5000 after retraining on pseudo-labeled OOD documents, the pipeline remains a theoretical design document.

2. **Synthetic-only OOD evaluation** (unanimous): The 13-model consensus from EXP-009 unanimously warned that synthetic AUROC likely overestimates real-world detection. Evaluation on Tobacco800, RVL-CDIP, CORD, or handwritten forms is the highest-priority future work.

3. **Calibration not demonstrated on VLM teachers** (unanimous): The pipeline's core use case --- calibrating VLM pseudo-labels --- is described as "planned" but not executed. The demonstrated calibration is on SigLIP2, which uses a different output distribution and operates in-distribution.

4. **Dead code thresholds**: The sigma-squared (0.64) and entropy (1.2) thresholds never trigger on actual data, effectively reducing the multi-threshold gating system to a pure Mahalanobis distance check.

---

## Actionable Suggestions for Improvement

### Critical (must address before resubmission)

1. **Execute one full expansion cycle**: Pseudo-label 500-1,000 OOD documents, retrain SigLIP2, report: (a) DIQA-5000 ID SRCC maintenance (target > 0.90), (b) OOD improvement metrics, (c) drift diagnostics across the iteration.

2. **Demonstrate VLM calibration on VLM outputs**: Run Gemini 3 Flash and GPT-4.1 on 3,500 DIQA-5000 training images, fit per-model per-dimension isotonic calibration, evaluate on test set and at least one real OOD dataset. Report reliability curves.

3. **Evaluate on real-world OOD datasets**: Test the Mahalanobis detector and VLM teachers on Tobacco800, RVL-CDIP, or CORD. Report AUROC and VLM SRCC on naturally-occurring OOD documents.

### Important (significantly strengthens paper)

4. **Add pipeline ablation study**: OOD gate on/off, single-teacher vs. dual-teacher vs. dual+tiebreaker, and weighting schemes. Quantify how each component affects downstream student performance.

5. **Reframe the "14x MAE reduction" claim**: Explicitly state this results from a [0,1] to [1,5] scale mismatch, not a difficult calibration problem. Consider reporting the meaningful metric (post-calibration MAE ~0.17 on MOS scale).

6. **Address dead code thresholds**: Either validate data-calibrated thresholds (sigma-squared p75 = 0.072, entropy p75 = 0.625) or simplify the gating to pure Mahalanobis distance and remove the unused thresholds.

### Minor (polish)

7. **Fix Section 6.3 numbering**: Replace repeated "1." with incrementing numbers in medium-term and longer-term lists.

8. **Standardize model naming**: Use consistent "Gemini 3 Flash Preview" or "Gemini 3 Flash" throughout.

9. **Clarify Appendix B table headers**: "d_M OOD" -> "d_M Tier2"; specify meaning of "sigma_sq Auto".

10. **Specify exact prompts and parsing rules**: Include the JSON prompt template, parse failure handling, and model version strings for reproducibility.

11. **Tighten Tier 2 decision policy**: Reconcile the Tier 2 description (Qwen3-VL-8B cross-validation) with the teacher section (Gemini/GPT-4.1/Claude Haiku). Clarify which models run at which stage.

12. **Define MainScore vs. wSRCC explicitly**: When mixing these metrics across tables, include the formula (wSRCC = 0.5 * SRCC_overall + 0.25 * SRCC_sharpness + 0.25 * SRCC_color) at first use.

---

## Areas of Disagreement Among Reviewers

| Topic | Range | Notes |
|-------|-------|-------|
| Technical Soundness | 3-5 | GPT-5.2 penalizes for unproven end-to-end claims; Grok rates methods as individually rigorous |
| Completeness | 2-4 | Largest spread; GPT-5.2 treats missing end-to-end as central; Grok treats it as important but not fatal |
| Reproducibility | 3-5 | GPT-5.2 wants exact prompts/hyperparameters; Grok finds artifact documentation exceptional |
| Novelty | 3-4 | GPT-5.2 sees mostly known components; others credit domain-specific synthesis as genuinely novel |

The disagreements are primarily about severity, not direction. All reviewers identify the same issues; they differ on how much each gap reduces the paper's contribution.

---

## Overall Assessment

Paper 7 presents a well-designed, thoroughly documented pseudo-labeling pipeline that addresses a genuine bottleneck in document IQA. The component-level validation is strong, the writing is excellent, and the limitations are honestly reported. However, the paper's central claim --- that iterative pseudo-labeling safely expands domain coverage --- remains unproven. Executing one complete expansion cycle, demonstrating VLM calibration on actual VLM outputs, and evaluating on real-world OOD documents would transform this from a promising design document into a validated system paper.

**Consensus verdict: Major Revision, with high confidence that addressing items 1-3 above would elevate the paper to Accept.**
