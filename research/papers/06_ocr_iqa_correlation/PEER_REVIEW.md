# Peer Review: Paper 6 --- OCR-IQA Correlation
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
**Note**: DeepSeek V3 was unavailable (model ID not found on OpenRouter). Review synthesized from 4 of 5 models.

---

## Consensus Recommendation: Minor Revision

Three of four reviewers recommend **Minor Revision**; one (GPT-5.2) recommends **Major Revision**. The consensus is Minor Revision with specific statistical and experimental improvements required before the claims are fully robust.

## Criteria Scores (Averaged Across 4 Reviewers)

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | Mean |
|-----------|---------|----------------|-----------|---------------|------|
| Technical Soundness | 3 | 4 | 4 | 4 | **3.75** |
| Completeness | 3 | 4 | 3 | 3 | **3.25** |
| Clarity | 4 | 5 | 4 | 4 | **4.25** |
| Novelty | 3 | 4 | 4 | 4 | **3.75** |
| Reproducibility | 4 | 5 | 5 | 5 | **4.75** |

## Points of Agreement (All 4 Reviewers)

1. **Reproducibility is excellent.** Deterministic seeds (`base_seed + image_idx * 100 + tier_idx`), complete environment specifications, artifact paths, and cost breakdown (~$3.80 total) set a gold standard for arXiv technical reports.

2. **Paired analysis is a methodological strength.** The delta-CER / delta-MOS design controlling for per-document complexity is well-motivated and produces stronger, more interpretable correlations.

3. **MOS scale compression is a significant concern.** The DeQA MOS spans only 0.412 points (2.942-3.354) on a 1-5 scale across all quality tiers. This limits practical thresholding and suggests the model may not fully capture document-specific degradation severity.

4. **Synthetic-only distortions limit external validity.** All distortions are generated via Augraphy + Albumentations. Real-world degradations (camera capture, photocopying, fax, physical aging) may produce different quality-CER relationships. This is the most frequently cited limitation.

5. **IQA baseline comparisons are missing.** The paper claims DeQA-Doc has downstream validity but does not compare against traditional NR-IQA methods (BRISQUE, NIQE, MUSIQ) or simple heuristics (blur/noise/skew measures). Without baselines, it is unclear whether DeQA-Doc offers additional predictive value over simpler alternatives.

6. **High baseline CER restricts dynamic range.** CER of 0.28-0.52 on undistorted originals (due to FUNSD form complexity) compresses the available range for quality-CER correlation, potentially underestimating the relationship on cleaner document types.

7. **The "catastrophic failure plateau" and PRISTINE-to-HIGH boundary findings are valuable.** All reviewers recognized these as actionable insights for production quality gating.

## Points of Disagreement

### Statistical Independence (GPT-5.2 vs. Others)

GPT-5.2 raised a critical concern that the unpaired correlations treat 1,200 images as i.i.d. when they are actually 6 variants of 200 base documents, potentially inflating p-values. GPT-5.2 recommends cluster-bootstrapping by document or mixed-effects modeling. The other three reviewers did not flag this issue, though it is a valid statistical concern.

**Resolution**: This is a legitimate methodological point. While the paired analysis (Section 4.2) already addresses within-document dependence, the unpaired analysis in Table 1 should either (a) report cluster-bootstrapped confidence intervals, or (b) explicitly acknowledge the non-independence limitation. Given the extremely small p-values (< 10^-56), the qualitative conclusions would likely survive correction, but the exact p-values should not be taken at face value.

### Monotonicity Claim (GPT-5.2)

GPT-5.2 noted that the monotonicity claim (Section 4.3) is contradicted by the data: Tesseract CER at LOW (0.819) > DEGRADED (0.811), and MOS at LOW (2.942) < DEGRADED (2.947). This is a valid observation that should be reconciled in the text.

**Resolution**: The non-monotonicity at LOW/DEGRADED is small and non-significant (Table 4 shows p = 0.819 for Tesseract), consistent with the "catastrophic failure plateau" interpretation. The text should qualify the monotonicity claim as "approximately monotonic" or "monotonic through the HIGH tier."

### Severity of Revision Required

GPT-5.2 rates Technical Soundness at 3/5 and recommends Major Revision primarily due to the statistical independence issue and missing baselines. The other three reviewers rate Technical Soundness at 4/5 and consider these addressable with minor additions.

## Top 3 Strengths (Consensus)

1. **Rigorous controlled experimental design** with deterministic distortion pipeline, paired analysis controlling for document complexity, and multi-engine evaluation across open-source and commercial OCR systems.

2. **Actionable production insights**: the PRISTINE-to-HIGH quality boundary (MOS ~ 3.07) as a universal quality gate, engine-specific sensitivity profiles for cost-quality routing, and the catastrophic failure plateau informing rejection policies.

3. **Exemplary reproducibility**: complete artifact paths, deterministic seeds, package versions, compute costs, and data licensing documentation that exceeds typical arXiv standards.

## Top 3 Weaknesses (Consensus)

1. **No comparison against baseline IQA methods** (BRISQUE, NIQE, MUSIQ, or simple heuristics). Without baselines, the paper cannot establish whether DeQA-Doc's learned quality scores offer incremental predictive value over cheaper alternatives.

2. **Synthetic-only evaluation** limits generalizability claims. The augmentation pipeline (Augraphy + Albumentations) may not faithfully represent real-world document degradation patterns, and correlation strengths could differ on naturally degraded documents.

3. **MOS scale compression** (0.412 range on a 1-5 scale) undermines practical thresholding and suggests the quality model may need document-specific calibration to be operationally useful.

## Specific Suggestions for Improvement

### Required (Consensus)

1. **Add IQA baseline comparisons.** Run BRISQUE and/or NIQE on the same 1,200 images. Compute their CER correlations. This directly tests whether a learned DIQA model outperforms traditional no-reference IQA for downstream prediction. (Low effort: models are available in `pyiqa` or OpenCV.)

2. **Address statistical non-independence.** Either (a) add cluster-bootstrapped confidence intervals (bootstrapping at the document level, not the image level) for Table 1 correlations, or (b) add a clear methodological note explaining why the paired analysis (Table 2) is the primary result and unpaired correlations are supplementary.

3. **Qualify the monotonicity claim.** Table 3 shows non-monotonic behavior at LOW/DEGRADED for both CER and MOS. Revise Section 4.3 to state "approximately monotonic" and note that convergence at the tails is consistent with the catastrophic failure plateau.

4. **Propose a scale calibration.** Given the compressed MOS range, add a brief analysis of isotonic regression or percentile-based stretching to map MOS to a wider operational scale. This is already noted in the RESEARCH_AGENDA and would be low effort.

### Recommended (Majority)

5. **Add WER alongside CER.** Word error rate captures different failure modes (word boundaries, segmentation) and is already partially available in the dataset. (Qwen 3.5+, Grok 4.1 Fast)

6. **Include a born-digital control set.** A small set of born-digital PDFs with near-zero baseline CER would demonstrate the correlation under ideal conditions, isolating image quality from document complexity. (Gemini 3.1 Pro)

7. **Clarify the 0-1 vs 1-5 scale discrepancy.** Table in Section 3.1 uses a 0-1 "Target Quality Range" for distortion profiles while MOS is on a 1-5 scale. Add a note explaining this is the augmentation intensity parameter, not a quality score. (GPT-5.2)

### Desirable (Individual)

8. **Add non-linear regression analysis.** The PLCC < SRCC gap across all engines suggests a monotonic but non-linear relationship. Fitting logistic or polynomial regression would better characterize the functional form. (GPT-5.2)

9. **Precision-recall curves for quality gating.** For each engine, compute precision-recall for a binary "high CER" classifier using MOS thresholds. This would directly demonstrate the quality gating use case. (Gemini 3.1 Pro)

10. **Confirm DeQA-Doc weights availability.** Exact reproduction requires access to the DeQA-Doc-3Specialists model weights. Clarify whether these are publicly available or provide a reference. (GPT-5.2)

## Minor Issues

- **Figure references**: Line 165 references "Figure 4" but no figures are included in the manuscript. Either add the figures or remove the reference.
- **Series context assumptions**: References to "Reports 1-5" and "Report 7" assume familiarity with the series. Add brief standalone descriptions for readers encountering this paper independently.
- **DeQA-Score citation**: The arXiv identifier in the References section contains a placeholder (`arXiv:2401.xxxxx`). Replace with the actual identifier.
- **Google Vision CER discrepancy**: Line 169 mentions a 0.006 percentage point difference between ORIGINAL and PRISTINE for Google Vision, but Table 3 shows identical values (0.284). Reconcile or clarify.

## Per-Model Review Summaries

### GPT-5.2 (Major Revision)
Most critical reviewer. Focused on statistical rigor: non-independence of clustered observations, inflated p-values, need for mixed-effects modeling. Also highlighted the monotonicity contradiction and the 0-1 vs 1-5 scale confusion in the tier table. Scored novelty lowest (3/5), viewing the contribution as useful but not entirely new.

### Gemini 3.1 Pro (Accept)
Most favorable reviewer. Praised the paper as "gold standard" for reproducibility and highlighted the practical value of treating DIQA as a downstream task predictor. Gave the highest clarity score (5/5). Main concern was synthetic-only evaluation and MOS compression. Suggested born-digital control set and scale calibration.

### Qwen 3.5+ (Minor Revision)
Balanced assessment. Agreed with GPT-5.2 on the need for IQA baselines and with Gemini on reproducibility strength. Uniquely noted the missing WER analysis and the figure reference issues. Scored completeness lowest (3/5) due to limited document diversity and missing baseline comparisons.

### Grok 4.1 Fast (Minor Revision)
Focused on practical value and implementation feasibility. Praised deterministic seeds, appropriate statistics, and actionable quality gating thresholds. Aligned with other reviewers on synthetic-only and MOS compression concerns. Emphasized the high user value for document processing pipelines.

### DeepSeek V3 (Unavailable)
Model ID `deepseek/deepseek-v3-0324` was not found on OpenRouter. This review slot was not filled.
