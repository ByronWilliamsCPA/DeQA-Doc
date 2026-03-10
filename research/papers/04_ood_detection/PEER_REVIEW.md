# Peer Review: Paper 4 — OOD Detection
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3 (deepseek/deepseek-v3-0324) was unavailable on OpenRouter during this review. The consensus reflects 4 of 5 panelists. DeepSeek V3.2 was attempted as a substitute but the tool retained the original model ID.

## Consensus Recommendation: Minor Revision

**Vote tally**: 3 Minor Revision (Gemini, Qwen, Grok) / 1 Major Revision (GPT-5.2)

The paper presents a well-engineered, practically valuable embedding-space OOD detector for document quality pipelines. The method is technically sound, the writing is exceptionally clear, and reproducibility is exemplary. However, the reliance on synthetic-only OOD evaluation, the absence of baseline OOD method comparisons, and the lack of downstream impact measurement prevent the paper from fully supporting its deployment-readiness claims.

## Aggregated Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 5 | **4.0** |
| Completeness | 2 | 3 | 3 | 3 | **2.75** |
| Clarity | 4 | 5 | 4 | 5 | **4.5** |
| Novelty | 3 | 3 | 3 | 4 | **3.25** |
| Reproducibility | 4 | 5 | 5 | 5 | **4.75** |

**Confidence scores**: GPT-5.2: 8/10, Gemini 3.1 Pro: 9/10, Qwen 3.5+: 7/10, Grok 4.1 Fast: 9/10

## Points of Agreement (Unanimous)

### Strengths
1. **Near-zero marginal cost via embedding reuse** (1-2ms latency, no additional forward pass). All reviewers highlighted this as the paper's most compelling practical contribution.
2. **Excellent reproducibility**. Artifact paths, seeds, compute costs, API code examples, and threshold values are all provided. Scored 4-5/5 by all reviewers.
3. **Transparent limitation discussion**. The paper candidly acknowledges synthetic-only evaluation, small sample sizes, and the circular training dependency. Multiple reviewers noted this increases credibility.
4. **Mahalanobis distance dominates aleatoric signals**. The threshold sensitivity analysis showing 93.7% vs 68.2% auto-accept rates is well-demonstrated and practically actionable.

### Weaknesses
1. **Synthetic-only OOD evaluation** is the dominant threat to validity. All 4 reviewers flagged this as the primary weakness. AUROC = 0.9963 on programmatically generated documents likely overestimates real-world performance. Real-world OOD documents exhibit more diverse and subtle distribution shifts.
2. **No baseline OOD method comparisons**. The paper does not evaluate cosine distance, KNN distance, energy-based scores, one-class SVM, isolation forest, or Gaussian mixture models on the same embeddings. Without baselines, it is impossible to attribute performance to Mahalanobis + Ledoit-Wolf specifically.
3. **No downstream impact evaluation**. The paper measures detection metrics (AUROC, TPR/FPR) but does not show whether flagging OOD documents actually improves quality prediction accuracy (MAE, SRCC) on auto-accepted images.
4. **Small per-category sample sizes** (n=20-30). Per-category AUROC confidence intervals are wide. A category with 30 samples and 100% detection has a 95% CI of approximately [88.4%, 100%] by Clopper-Pearson. The "perfect AUROC" claims for 10/13 categories are not statistically robust.

### Suggestions for Improvement (Consensus)
1. **Add real-world OOD benchmarks** (RVL-CDIP, Tobacco800, CORD). Report AUROC, AUPRC, and FPR@95TPR with clear ID vs OOD protocol.
2. **Add baseline OOD detectors** on the same 768-dim embeddings: cosine distance to centroid, Euclidean distance, KNN (k=1/5/10), GMM (2-10 components), one-class SVM, isolation forest. Include latency comparison.
3. **Report bootstrap confidence intervals** for AUROC (overall and per-category). Include binomial CIs for detection rates as standard table columns.
4. **Measure downstream impact**: show SigLIP2-IQA prediction MAE on auto-accepted vs flagged/rejected subsets. A "MAE vs Mahalanobis distance decile" plot would connect detection metrics to user value.
5. **Increase OOD sample sizes** to at least n=50-100 per category for statistically meaningful per-category claims.

## Points of Disagreement

### Technical Soundness (Range: 3-5)
- **GPT-5.2 (3/5)**: Concerned about limited statistical reporting (no CIs for AUROC, no uncertainty on operating-point metrics) and the FPR "by construction" claim being misleading given the train-test distance shift.
- **Grok 4.1 Fast (5/5)**: Viewed the Mahalanobis + Ledoit-Wolf framework as fully valid, correctly applied, with adequate CI awareness (the paper mentions Clopper-Pearson bounds in Section 5.3).
- **Resolution**: The method itself is sound and standard. The disagreement is about the completeness of statistical reporting, not the validity of the approach. Adding bootstrap CIs would satisfy both perspectives.

### FPR "by construction" Claim
- **GPT-5.2** flagged the claim that FPR is "~5% by construction" (at threshold = train+val p95) as misleading, since the test distribution is shifted (test p95 = 48.5 vs train+val p95 = 30.8), meaning the actual test FPR is much higher than 5%.
- **Other reviewers** noted this concern less prominently but acknowledged the train-test gap.
- **Resolution**: The paper should explicitly report the achieved test-set FPR at the train+val p95 threshold, not imply it is 5% by construction. A small table mapping threshold source to achieved (TPR, FPR) on test would resolve this.

### Novelty (Range: 3-4)
- **Grok 4.1 Fast (4/5)**: Viewed the domain-first application to IQA regression pipelines as meaningfully novel.
- **Others (3/5)**: Considered the core methodology (Lee et al. 2018 Mahalanobis) as well-established, with novelty primarily in application rather than algorithmic innovation.

## Minor Issues (Consolidated)

1. **Table 3 redundancy**: "Current (v1)" and "d_M Only" rows show identical metrics (93.7% auto-accept for both). Clarify that this demonstrates the v1 aleatoric thresholds are effectively disabled, or consolidate into a single row with annotation. (Flagged by 3/4 reviewers)
2. **Figure references**: Lines 124, 144, 174, 196 reference Figures 1, 2, 4 that are not embedded in the markdown format. Either embed or note they are available in the full PDF.
3. **Embedding invertibility claim**: Line 322 states embeddings "cannot be inverted to reconstruct original images." This is stronger than typically defensible. Soften to "are not intended to be invertible; reconstruction is non-trivial." (GPT-5.2)
4. **Section numbering**: Section 6 "Conclusion" is followed by Section 7 "Reproducibility," but the introduction states "Section 6 concludes." Update to reflect Section 7 or restructure. (Qwen 3.5+)
5. **"EXP-009" reference**: The 13-model consensus analysis is mentioned (line 218, 326) without sufficient methodology description for a standalone paper. Add a brief description or formal citation. (Qwen 3.5+)
6. **Acknowledgments tone**: "conducted via OpenRouter API across 13 frontier LLMs" reads informal for an academic acknowledgment. Consider specifying methodology or models used. (Gemini 3.1 Pro)
7. **"Effective N" column** in Table 3 (line 178) needs a definition. (Qwen 3.5+)

## Reviewer-Specific Notable Points

### GPT-5.2 (Most Critical)
- Strongest emphasis on the FPR reporting issue and the need for explicit test-set FPR at each threshold.
- Suggested documenting synthetic OOD generation scripts, fonts, and parameter ranges for reproducibility.
- Recommended a "MAE vs d_M decile" plot to demonstrate downstream utility.

### Gemini 3.1 Pro (Most Positive on Writing)
- Gave Clarity 5/5 and Reproducibility 5/5 — the highest marks on both.
- Highlighted the tiered routing design as "remarkably cohesive" systems engineering.
- Noted the single Gaussian assumption may mask subtle distribution shifts as the pipeline scales.

### Qwen 3.5+ (Most Detailed Minor Issues)
- Identified the most minor formatting and reference issues (5 specific items).
- Emphasized figures not visible in markdown format.
- Noted the paper is the first application of embedding-space OOD detection to IQA (line 60) as a meaningful but incremental contribution.

### Grok 4.1 Fast (Most Positive Overall)
- Gave Technical Soundness 5/5 — the only reviewer to do so.
- Strongest endorsement of the two-tier routing architecture for production use.
- Noted the retrain protocol (Section 5.2) as a strength for long-term system evolution.

## Priority Action Items

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Validate on real-world OOD datasets (RVL-CDIP, Tobacco800, CORD) | Medium | High — resolves the primary weakness unanimously identified |
| P0 | Add baseline OOD method comparisons on same embeddings | Low | High — contextualizes Mahalanobis performance |
| P1 | Report bootstrap CIs for AUROC and binomial CIs for detection rates | Low | Medium — strengthens statistical claims |
| P1 | Clarify FPR reporting: explicit test-set FPR at each threshold | Low | Medium — resolves ambiguity flagged by GPT-5.2 |
| P1 | Add downstream impact evaluation (MAE on auto-accepted vs flagged) | Low-Medium | Medium — connects detection to user value |
| P2 | Increase OOD sample sizes to n=50-100 per category | Medium | Medium — narrows confidence intervals |
| P2 | Resolve Table 3 redundancy and minor formatting issues | Low | Low — polish |
| P2 | Document synthetic OOD generation scripts and parameters | Low | Low — improves reproducibility |
