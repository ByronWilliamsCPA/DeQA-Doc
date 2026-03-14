# Peer Review: Paper 8 — SigLIP2-IQA Training
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3-0324 was unavailable on OpenRouter during this review session (invalid model ID). The consensus below reflects 4 of 5 panelists. All other models responded successfully.

## Consensus Recommendation: Minor Revision

**Vote breakdown**: 1 Major Revision (GPT-5.2), 3 Minor Revision (Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast)

The panel unanimously recognizes the paper's strong practical contributions -- replacing a 28B-parameter MLLM ensemble with an 86.6M-parameter ViT that runs 30x faster while achieving higher single-model accuracy. The consensus is that the paper is suitable for publication after addressing metric inconsistencies, adding targeted ablation studies, and clarifying reproducibility details.

---

## Aggregate Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 5 | **4.00** |
| Completeness | 3 | 3 | 3 | 4 | **3.25** |
| Clarity | 3 | 5 | 4 | 5 | **4.25** |
| Novelty | 3 | 4 | 3 | 4 | **3.50** |
| Reproducibility | 2 | 4 | 3 | 5 | **3.50** |
| **Overall** | | | | | **3.70** |

---

## Consensus Strengths (Agreed by 3+ Reviewers)

### 1. Exceptional Compute-to-Efficiency Ratio (4/4 agree)
Achieving +24% performance over a 21B+ parameter ensemble using only 86.6M parameters fundamentally alters operational cost frameworks for document processing pipelines. The 30x inference speedup (~100ms vs ~3,000ms) makes real-time quality routing practical.

### 2. Dual-Purpose Architecture (4/4 agree)
Generating calibrated IQA scores alongside Mahalanobis OOD detection embeddings (AUROC = 0.9963) in a single ~100ms forward pass is an elegant systemic optimization that eliminates the need for a separate OOD model.

### 3. Uncertainty-Aware Design (4/4 agree)
The integration of Gaussian NLL loss produces operationally useful sigma-squared values with clear thresholds (0.64 for auto-accept), enabling confidence-based routing without ensembles or Monte Carlo dropout. The 93.7% auto-accept rate demonstrates practical utility.

### 4. Well-Motivated Architecture Selection (3/4 agree)
NaFlex resolution handling is well-motivated for documents -- preserving aspect ratios and text legibility that fixed-resolution models destroy. The rationale for SigLIP2 over alternatives is clearly articulated.

---

## Consensus Weaknesses (Agreed by 3+ Reviewers)

### 1. Missing Ablation Studies (4/4 agree) -- CRITICAL
All four reviewers flagged the absence of ablation studies as the primary weakness. The paper makes multiple design claims that are not experimentally isolated:

- **Two-phase training**: No comparison of single-phase vs two-phase, or Phase 1 duration variants (5/10/15/20 epochs)
- **Patch count**: No comparison of 576 vs 784 patches on sharpness SRCC
- **Dropout rate**: No justification for 0.3 vs lower values
- **Multi-task regularization**: Section 6.3 claims multi-task heads provide "implicit regularization" but this is never verified by comparing IQA-only vs multi-task training
- **Loss components**: No ablation of NormInNorm-only vs NormInNorm + GaussianNLL vs PCGrad impact

### 2. Metric Inconsistency (4/4 agree) -- MUST FIX
The abstract reports wSRCC = 0.886 (Line 14), but Table 5.1 reports weighted SRCC = 0.891 (Line 212), and the baseline comparison table uses 0.886 (Line 222). This internal contradiction undermines confidence in reported results. The paper must:
- Establish a single canonical wSRCC value
- Clarify whether the difference reflects pre- vs post-calibration
- Ensure all instances are consistent throughout

### 3. Severe OOD Degradation (4/4 agree)
MainScore drops from 0.886 to 0.620 on synthetic OOD documents (30% degradation). While honestly reported, the paper would benefit from:
- Deeper analysis of which OOD categories fail most (binarized, non-Latin, extreme DPI)
- Whether uncertainty estimates (sigma-squared) correlate with OOD prediction errors
- Comparison of OOD degradation rates vs baselines

### 4. Multi-Task Heads Claimed But Not Evaluated (3/4 agree)
Section 6.3 lists five additional task heads (script detection, source detection, orientation, shadow severity, warping severity) but provides zero accuracy metrics for any of them. This section should either include performance tables or be removed.

---

## Individual Reviewer Highlights

### GPT-5.2 (Most Critical -- Major Revision)
- Flagged MAE reporting confusion: raw MAE of 2.424 is confusing if outputs are on [0,1] scale (Line 245 vs Line 214)
- Noted missing weight decay, beta values, augmentation magnitudes, and random seeds
- Wanted bootstrap CI for wSRCC (not just per-dimension), and details on stratification
- Suggested clearer "same split / same metric / same calibration" guarantees for all baseline comparisons

### Gemini 3.1 Pro (Favorable -- Minor Revision)
- Praised writing as "exceptionally clear, perfectly structured"
- Highlighted industry value: "exactly what the industry desperately needs"
- Suggested qualitative visual examples of the "extreme quality blindspot" (prediction range [0.32, 3.92])
- Noted missing inference hardware specs for baseline MLLMs in Table 5.2

### Qwen 3.5+ (Moderate -- Minor Revision)
- Noted the comparison of single fine-tuned model vs ensemble "under different conditions" requires clearer framing
- Wanted empirical backbone comparison (DINOv2, CLIP vs SigLIP2), not just the rationale table
- Flagged private Modal volume checkpoint and unclear DIQA-5000 licensing as reproducibility barriers
- HyperIQA++ footnote marker placement is ambiguous

### Grok 4.1 Fast (Most Favorable -- Minor Revision, near Accept)
- Gave highest scores across all criteria (5/4/5/4/5)
- Praised technical feasibility, industry alignment, and low training cost (~$5)
- Noted ONNX exportability as a long-term deployment advantage
- Only significant critique: missing ablation on training phases and NaFlex patches

---

## Actionable Suggestions for Improvement

### Priority 1: Must Fix Before Resubmission

1. **Resolve wSRCC inconsistency**: Determine whether 0.886 or 0.891 is correct. Update abstract, Section 5.1, and Table 5.2 to use a single canonical value. Explicitly state whether this is computed from per-dimension SRCCs as defined in Section 2.1 (Line 73).

2. **Fix MAE reporting**: Clarify how "raw MAE 2.424" arises if model outputs are on [0,1] scale. State the prediction scale at each stage and compute MAE consistently. The calibration table (Section 5.3) should clearly distinguish raw output scale from MOS scale.

3. **Add ablation table** (minimum viable):
   - Single-phase vs two-phase training (validates warmup rationale)
   - 576 vs 784 patches on sharpness SRCC (validates resolution bottleneck hypothesis)
   - With vs without GaussianNLL loss (validates uncertainty component)

### Priority 2: Strongly Recommended

4. **Evaluate or remove multi-task heads**: Either add accuracy metrics for script/source/orientation/shadow/warping heads (Section 6.3), or remove the section entirely. If keeping, add an IQA-only vs multi-task ablation to substantiate the regularization claim.

5. **Strengthen reproducibility**: Provide public access path for model checkpoint (HuggingFace or GitHub release). State DIQA-5000 licensing and availability explicitly. Add exact optimizer parameters (weight decay, betas) and random seeds.

6. **Deepen OOD analysis**: Break down the 0.886 -> 0.620 degradation by OOD category. Show whether sigma-squared correlates with prediction error magnitude on OOD data. Compare OOD degradation rates across baselines.

### Priority 3: Nice to Have

7. **Add figures**: Uncertainty calibration plots (sigma-squared vs absolute error), qualitative failure examples, attention map visualizations for sharpness vs color heads.

8. **Clarify baseline fairness**: Specify inference hardware for all baselines in Table 5.2. Note whether HyperIQA++ score is from same DIQA-5000 split or synthetic OOD.

9. **Formalize internal references**: Convert "Paper 4", "Paper 5" etc. to proper citations if publishing as standalone arXiv report.

10. **Add statistical reporting**: Report bootstrap CI for wSRCC (not just per-dimension). Consider multi-seed runs (3+) to demonstrate stability.

---

## Minor Issues

| Location | Issue | Suggested Fix |
|----------|-------|---------------|
| Line 14 vs 212 vs 222 | wSRCC values inconsistent (0.886 / 0.891 / 0.886) | Unify to single canonical value |
| Line 214 vs 245 | MAE "raw model output scale" vs "Predictions on [0,1] scale" | Clarify scale at each reporting stage |
| Line 248 | "wSRCC 0.891 vs 0.891; wMAE 0.174 vs 0.173" hard to parse | Rewrite for clarity |
| Line 225 | HyperIQA++ footnote marker ambiguous | Clarify what the asterisk applies to |
| Line 300-301 | Shadow/Warping heads listed but never evaluated | Add metrics or remove |
| Line 350 | Private Modal volume limits reproducibility | Add public access path |
| Lines 49-53 | Internal paper references ("Paper 4", "Paper 5") | Convert to formal citations for standalone publication |
| Table 5.2 | Missing inference hardware specs for MLLM baselines | Add GPU type for all entries |
| References | Some citations missing venue/year (e.g., RichIQA) | Complete all reference entries |

---

## Reviewer Confidence

| Reviewer | Confidence | Notes |
|----------|-----------|-------|
| GPT-5.2 | 7/10 | Limited by inability to verify underlying code/data |
| Gemini 3.1 Pro | 9/10 | High confidence in methodology and industry trends |
| Qwen 3.5+ | -- | Did not provide explicit confidence score |
| Grok 4.1 Fast | 9/10 | Minor uncertainty on external baseline verification |
| DeepSeek V3 | N/A | Model unavailable |

---

## Summary

Paper 8 presents a compelling case for compact, specialized vision transformers as replacements for large MLLM ensembles in document image quality assessment. The 30x speedup, dual-purpose embeddings, and calibrated uncertainty estimation represent genuine practical advances. The consensus recommendation of **Minor Revision** reflects confidence in the core contribution, with the primary revision requirements being: (1) fixing the wSRCC metric inconsistency, (2) adding targeted ablation studies to substantiate design claims, and (3) improving reproducibility through public artifact access. These are addressable without fundamental changes to the paper's structure or conclusions.
