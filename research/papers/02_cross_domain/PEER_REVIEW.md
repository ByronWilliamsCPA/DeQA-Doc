# Peer Review: Paper 2 — Cross-Domain Generalization
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3 was unavailable during review (model ID `deepseek/deepseek-v3-0324` not found on OpenRouter). Consensus is based on 4 of 5 reviewers. All reviewers used neutral stance.

## Consensus Recommendation: Minor Revision

**Vote tally**: 3 Minor Revision (Gemini, Qwen, Grok) / 1 Major Revision (GPT-5.2)

The paper is a well-executed empirical study with clear practical value. The primary revision requirements center on statistical rigor and ground truth validation rather than fundamental methodological concerns.

---

## Consensus Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 5 | **4.0** |
| Completeness | 3 | 4 | 4 | 4 | **3.75** |
| Clarity | 4 | 5 | 4 | 5 | **4.5** |
| Novelty | 4 | 4 | 4 | 4 | **4.0** |
| Reproducibility | 3 | 4 | 5 | 5 | **4.25** |
| **Overall** | **3.4** | **4.2** | **4.2** | **4.6** | **4.1** |

---

## Points of Agreement (All 4 Reviewers)

### Strengths

1. **Comprehensive model coverage**: Evaluating 7 VLMs, 3 fine-tuned models, and 5 NR-IQA baselines across 13 OOD categories provides exceptionally broad empirical grounding.

2. **Actionable pseudo-labeling triage**: The Tier A/B/C categorization (Section 5.2) maps directly from experimental findings to engineering decisions. Multiple reviewers highlighted this as the paper's most valuable practical contribution.

3. **Transparent limitations**: The paper honestly acknowledges synthetic ground truth constraints, small per-category sample sizes, and API snapshot dependency. This builds trust in the reported findings.

4. **Clear writing and organization**: All reviewers rated clarity at 4/5 or above. The logical flow from research questions through results to practical implications is well-executed.

### Weaknesses

1. **Synthetic ground truth unvalidated by human annotations**: All 4 reviewers flagged that ground truth MOS derived from generation parameters (line 68) may not align with human perceptual quality. This is the single most critical concern. Models that track generation parameters rather than perceived quality could appear to generalize better than they actually do.

2. **Missing confidence intervals / statistical significance**: No bootstrap CIs or hypothesis tests are provided for SRCC values. With per-category n = 20--30, individual category SRCC estimates have wide confidence intervals (~+/- 0.3). Model-to-model differences within categories cannot be declared significant.

3. **Small per-category sample sizes**: All reviewers noted that n = 20--30 limits the statistical power of per-category conclusions. The aggregate signal across 13 categories is robust, but individual category findings should be treated as approximate.

---

## Points of Disagreement

### Technical Soundness (Range: 3--5)

- **GPT-5.2 (3/5)**: Stricter evaluation focused on synthetic MOS validity risk, parse failure bias, and ID/OOD column inconsistencies in Table 1. Argued that conclusions about "generalization" may reflect alignment to generation parameters rather than true perceptual quality.
- **Grok 4.1 Fast (5/5)**: Most generous, viewing the methods as valid with standard IQA metrics and transparent acknowledgment of limitations. Considered statistical limitations adequately disclosed.
- **Resolution**: The truth likely falls between these positions. The methods are standard and correctly applied, but the synthetic ground truth introduces a systematic validity threat that limits the strength of conclusions. Score 4/5 is appropriate.

### Reproducibility (Range: 3--5)

- **GPT-5.2 (3/5)**: Flagged missing VLM prompt template, synthetic generation code, and parameter-to-MOS mapping as blocking replication.
- **Grok 4.1 Fast & Qwen 3.5+ (5/5)**: Praised the data archival (JSONL checkpoints, metrics files, script paths) as enabling replication.
- **Resolution**: Data archival is strong, but the exact prompt text and synthetic generation recipe are indeed missing. These are critical for VLM evaluation reproducibility. Score 4/5 is appropriate.

---

## Consolidated Suggestions for Improvement

### Priority 1: Must Address (Unanimous)

1. **Add bootstrap confidence intervals** for all SRCC/wSRCC values in Tables 1, 2, and Appendix A1. Report 95% CIs via bootstrap resampling (1000+ iterations). For per-category values at n = 20--30, explicitly state the CI width.

2. **Validate synthetic ground truth with human annotations**: Collect human MOS for a stratified subset of ~50--100 OOD images (especially from Tier C categories: binarized, extreme DPI, pristine). Report correlation between synthetic MOS and human MOS. If they diverge, qualify claims accordingly.

3. **Include the exact VLM prompt template** used for evaluation. Zero-shot VLM performance is highly sensitive to prompt wording. Append the full JSON-enforcing prompt as an appendix.

### Priority 2: Strongly Recommended (3+ Reviewers)

4. **Audit and report parse failures by category**: Test whether parse failures correlate with specific OOD categories or difficulty levels. Include a sensitivity analysis (e.g., worst/best-case imputation) rather than computing metrics on valid responses only.

5. **Resolve Table 1 ID/OOD comparability**: The "ID (DIQA-5000)" column mixes two different reference datasets (DIQA-5000 for VLMs, synthetic ID subset for fine-tuned models). Either separate into two columns or clearly rename to avoid ambiguity.

6. **Quantitatively test multi-model consensus**: The paper motivates consensus averaging (Section 5.1) but never tests it. Evaluate simple ensembles (mean/median rank aggregation) using existing checkpoint data and report whether combining GPT-4.1 + Gemini 3 Flash improves per-category SRCC.

### Priority 3: Recommended (1--2 Reviewers)

7. **Explain the Claude Haiku Tibetan/Myanmar anomaly**: Haiku achieves 0.833 on Myanmar but only 0.383 on Tibetan (Table 2). This large discrepancy for two non-Latin script categories warrants 1--2 sentences of explanation. *(Gemini)*

8. **Condense Sections 4.2--4.3**: The per-category and per-model analyses are thorough but dense. Consider moving detailed per-model commentary to an appendix and keeping only the tier-level summary in the main text. *(Qwen)*

9. **Soften causal claims**: Replace "This confirms that their quality judgments derive from general visual understanding" (line 150) with "This is consistent with quality judgments deriving from general visual understanding." Several claims are stated as confirmed when the synthetic benchmark provides suggestive rather than definitive evidence. *(GPT-5.2)*

10. **Include OOD detection integration results**: The paper references an embedding-space OOD detector (line 295) but does not show its impact on pseudo-labeling quality. A brief experiment would strengthen the practical narrative. *(Grok)*

---

## Minor Issues

| Issue | Source | Location |
|-------|--------|----------|
| Model naming inconsistency: "Gemini 3 Flash Preview" vs "Gemini 3 Flash" used interchangeably | GPT-5.2 | Lines 92 vs 133 |
| Table 1 footnote "*" for synthetic ID subset is easy to miss; consider clearer labeling | GPT-5.2, Gemini | Line 143 |
| "--" in Table 2 means "null SRCC" but the same notation in Appendix Table A1 lacks the footnote | Gemini | Lines 180 vs 368 |
| "Qwen3-VL-8B Think" abbreviation inconsistent with full name in table header | Grok | Line 381 |
| Future dates (March 2026) and unreleased model names may confuse readers if published as preprint | Grok | Lines 4, 342 |
| Table formatting: not all tables use consistent footnote styles | Grok | Various |

---

## Individual Reviewer Summaries

### GPT-5.2 (Major Revision)
The strictest reviewer. Core concerns: synthetic MOS validity as the foundational risk, parse failure bias (missing-not-at-random), and internal inconsistencies between Table 1 ID values and later discussion. Unique contribution: identified that claims about "general visual understanding" are too strong without human OOD validation. Confidence: 7/10.

### Gemini 3.1 Pro (Minor Revision)
Praised the paper as "exceptionally practical" with an actionable triage matrix. Distinguished between true model failures and "measurement ceilings" (e.g., pristine documents). Uniquely flagged Claude Haiku's Tibetan vs Myanmar anomaly as needing explanation. Confidence: 9/10.

### Qwen 3.5+ (Minor Revision)
Gave the highest reproducibility score (5/5), noting thorough data path and code documentation. Suggested sections 4.2--4.3 could be more concise. Recommended prompt ablation studies as an additional experiment. Confidence: 8/10.

### Grok 4.1 Fast (Minor Revision)
Most favorable review overall (mean 4.6/5). Praised the transparent error analysis and reproducibility package. Uniquely noted that future dates and model names may confuse preprint readers. Suggested including OOD detection integration results. Confidence: 8/10.

### DeepSeek V3 (Unavailable)
Model ID `deepseek/deepseek-v3-0324` was not recognized by OpenRouter. The closest available model (`deepseek/deepseek-v3.2`) was not substituted to preserve the requested panel composition.

---

## Summary Assessment

This paper makes a valuable empirical contribution to VLM-based document quality assessment. The identification of universal failure modes (binarized, extreme DPI, pristine), the specialization-generalization tradeoff finding, and the complementary model strengths analysis are all useful for the field. The primary limitations -- synthetic ground truth validation, statistical uncertainty quantification, and missing prompt documentation -- are addressable through targeted additions rather than fundamental redesign.

**Estimated effort for revision**: 1--2 weeks for Priority 1 items (bootstrap CIs, human validation subset, prompt appendix). Priority 2 items (parse failure audit, Table 1 cleanup, consensus testing) add another week. The paper should be publishable after these revisions.
