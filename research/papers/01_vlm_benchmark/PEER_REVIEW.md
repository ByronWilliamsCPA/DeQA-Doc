# Peer Review: Paper 1 -- VLM Benchmark
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
**Note**: DeepSeek V3 (deepseek/deepseek-v3-0324) was unavailable due to an invalid model ID on OpenRouter. Review synthesizes 4 of 5 panelists.

---

## Consensus Recommendation: Minor Revision

Three of four reviewers recommend Minor Revision or Accept with minor clarifications. One reviewer (GPT-5.2) recommends Major Revision, primarily due to concerns about parse-failure handling and incomplete metric definitions. The consensus leans toward **Minor Revision** -- the paper is technically sound and makes a genuine contribution, but several methodological details and missing analyses should be addressed before publication.

---

## Scores Summary

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | Mean |
|---|---|---|---|---|---|
| Technical Soundness | 4/5 | 4.5/5 | 4/5 | 5/5 | **4.4** |
| Completeness | 3/5 | 4/5 | 4/5 | 4/5 | **3.75** |
| Clarity | 4/5 | 5/5 | 4/5 | 5/5 | **4.5** |
| Novelty | 3/5 | 4/5 | 4/5 | 4/5 | **3.75** |
| Reproducibility | 3/5 | 4/5 | 5/5 | 5/5 | **4.25** |
| **Recommendation** | Major Rev. | Minor Rev. | Minor Rev. | Accept/Minor | **Minor Rev.** |

---

## Per-Reviewer Summaries

### GPT-5.2 (Most Critical)

**Recommendation:** Major Revision

**Key Concerns:**
- Computing metrics only on valid responses after nontrivial parse-failure rates (7.0% for Gemini 2.5 Pro) can bias results upward. Needs sensitivity analysis with worst-case imputation and intersection-set metrics.
- Synthetic OOD "ground truth MOS" derived from generation parameters is not equivalent to human perceptual MOS. Conclusions should be reframed or validated with human ratings on a subset.
- "MainScore" appears in Table 3 without definition in the metrics section.
- PLCC "4-parameter logistic curve fitting" is mentioned without sufficient fitting detail.
- Exact prompts are summarized but not reproduced verbatim, hampering replication.
- The 7-arm n=23 prompt optimization study has multiple-comparison risk that is not addressed.

**Strengths Noted:** Solid n=1,000 scale with uncertainty estimates; insightful ordinal/bias analysis; empirical demonstration of small-n prompt tuning pitfalls.

**Confidence:** 7/10

### Gemini 3.1 Pro (Most Positive on Clarity)

**Recommendation:** Accept with Minor Revisions

**Key Concerns:**
- Calibration is stated as mandatory but only discussed conceptually -- no baseline calibration curve is plotted.
- Chain-of-thought underperformance is noted but not explained. Extracting 1-2 CoT failure traces would illuminate why deliberation introduces noise.
- API temporal instability is a long-term reproducibility risk.

**Strengths Noted:** Statistical honesty in the n=23 vs n=1,000 finding; exceptional systematic bias and failure analysis; deeply actionable cost-performance guidance.

**Confidence:** 9/10

### Qwen 3.5+ (Highest Reproducibility Score)

**Recommendation:** Minor Revision

**Key Concerns:**
- Synthetic OOD ground truth is somewhat circular (parameter-derived).
- Missing ablation on prompt components (e.g., which parts of the rubric drive performance).
- Deeper analysis needed on why reasoning models underperform.

**Strengths Noted:** Exceptional data release (12,877 evaluations with reasoning); methodologically important small-sample finding; practical dual-model consensus recommendation.

**Confidence:** Not explicitly stated (estimated 8/10 from text)

### Grok 4.1 Fast (Most Positive Overall)

**Recommendation:** Accept with minor clarifications

**Key Concerns:**
- Lacks real-world OOD document evaluation (only synthetic).
- Variance estimation is entirely deferred to future work.
- Parse-failure sensitivity analysis would strengthen claims.

**Strengths Noted:** Rigorous methodology matching VQualA standards; high practical value for DIQA researchers; comprehensive data/script release for reproducibility.

**Confidence:** 8/10

---

## Points of Agreement (All 4 Reviewers)

1. **Small-sample prompt optimization finding is the paper's strongest methodological contribution.** The demonstration that n=23 optimization suggested +0.042 wSRCC that became -0.009 at full scale is broadly relevant beyond DIQA and should be emphasized.

2. **Data release is exemplary.** The 12,877 per-sample VLM evaluations with full reasoning traces set a high standard for VLM benchmarking transparency.

3. **Ordinal discrimination and bias analysis add substantial value** beyond standard correlation metrics. The finding that all VLMs systematically over-rate (64%-98% of images) while preserving rank order is important for pseudo-labeling applications.

4. **Synthetic OOD ground truth has limited external validity.** Parameter-derived MOS is not equivalent to human perceptual quality. Claims should be softened or validated.

5. **Chain-of-thought reasoning underperformance deserves deeper investigation.** All reviewers noted that the paper observes the phenomenon but does not adequately explain it.

6. **Writing quality is high.** All reviewers rated clarity 4/5 or 5/5.

---

## Points of Disagreement

| Issue | GPT-5.2 | Others |
|---|---|---|
| Parse-failure severity | Major concern -- biases all comparisons | Acknowledged but not blocking |
| Overall recommendation | Major Revision needed | Minor Revision sufficient |
| Reproducibility | 3/5 -- API drift and missing prompts are serious | 4-5/5 -- JSONL checkpoints adequately mitigate |
| Novelty | 3/5 -- incremental over Q-Doc | 4/5 -- structured JSON + newer models + scale are meaningful |

GPT-5.2 is the outlier, applying stricter standards around metric definitions and parse-failure methodology. The other three reviewers consider the same issues as addressable through minor additions rather than fundamental redesign.

---

## Consolidated Actionable Improvements

### Priority 1 (Required for Revision)

1. **Parse-failure sensitivity analysis.** Report metrics three ways: (a) excluding failures (current), (b) assigning dataset median to failures, (c) computing on intersection of images where all models returned valid JSON. Add a 1-paragraph sensitivity discussion.

2. **Define MainScore in Section 3.3.** Add the formula (MainScore = 0.5 * (PLCC_O + SRCC_O) + 0.25 * (PLCC_S + SRCC_S) + 0.25 * (PLCC_C + SRCC_C), or whatever the actual definition is) alongside the existing wSRCC definition.

3. **Include verbatim prompts in an appendix.** Reproduce the exact system prompt and user prompt template. Specify any JSON-repair or retry logic used.

4. **Fix category count inconsistency.** Abstract says "13 categories" but Section 3.1 describes 15. Reconcile.

5. **Standardize model naming.** Use consistent names throughout (e.g., always "Gemini 3 Flash Preview" or always "Gemini 3 Flash", not both).

### Priority 2 (Strongly Recommended)

6. **Add calibration demonstration.** Plot a scatterplot of raw VLM predictions vs. human MOS for the top model, and overlay a simple linear or isotonic regression calibration. Show post-calibration MAE. This concretely supports the "calibration is mandatory" claim.

7. **Analyze CoT failure traces.** Extract 1-2 examples from Qwen3-VL-8B Thinking where extended reasoning led to worse predictions. Does the model hallucinate defects? Over-weight irrelevant features? This would transform an observation into an insight.

8. **Specify PLCC fitting procedure.** State the 4-parameter logistic function form, fitting method (least squares), and whether fitting is done on the test set (standard practice in IQA but should be explicit).

### Priority 3 (Nice to Have)

9. **Add variance estimation pilot.** Run the top model (Gemini 3 Flash) 3-5 times with temperature > 0 on a 100-image subset. Report whether multi-sample standard deviation correlates with human annotation disagreement.

10. **Reframe synthetic OOD claims.** Either validate parameter-derived MOS against human ratings on a small subset, or explicitly state that OOD results measure "agreement with degradation parameters" rather than "perceptual quality correlation."

11. **Statistical framing for prompt experiments.** Note multiple-comparison risk in the 7-arm study. Consider reporting variance across several random stratified draws of n=23.

---

## Minor Issues

- Model naming inconsistency: "Gemini 3 Flash Preview" (abstract, Table in Section 3.2) vs. "Gemini 3 Flash" (Table 1 and throughout results).
- "Qwen 3.5 Flash" and "Qwen3-VL-8B" are different models but the naming convention makes this unclear. Consider a consistent format: Provider/Model/Size.
- Abstract says "13 categories" for OOD; Section 3.1 lists 15 categories. One of these is wrong.
- Table 3 introduces "MainScore" without prior definition.

---

## Overall Assessment

This is a well-executed empirical benchmark that makes a genuine contribution to the DIQA and VLM evaluation literature. The core finding -- that frontier VLMs approach supervised baselines (wSRCC 0.708 vs. 0.716) without any domain-specific training -- is supported by the data and has clear practical implications for pseudo-labeling pipelines. The small-sample optimization warning is methodologically important and broadly applicable.

The paper's main weaknesses are in completeness rather than soundness: calibration is discussed but not demonstrated, CoT underperformance is observed but not explained, and parse-failure handling could bias model comparisons. These are all addressable through targeted additions that do not require new experiments or fundamental restructuring.

**Consensus: Minor Revision.** Address Priority 1 items and at least items 6-8 from Priority 2 before submission.
