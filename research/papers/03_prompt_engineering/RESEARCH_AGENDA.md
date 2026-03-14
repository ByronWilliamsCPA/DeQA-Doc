# Research Agenda: Prompt Engineering for VLM-Based Quality Assessment

**Paper:** DeQA-Doc Technical Report 3/7
**Status:** Living document
**Last Updated:** 2026-03-08

---

## Confirmed Findings

### F1: Small-sample prompt optimization is unreliable
- **Evidence:** 7-arm pilot (n=23) predicted no-resize advantage of +0.042 wSRCC; full-scale (n=1000) showed -0.009.
- **Implication:** Minimum n=200 stratified samples recommended for prompt selection.
- **Confidence:** High. The reversal is unambiguous, and the statistical argument (SE ~0.08 at n=23) explains the mechanism.

### F2: Per-dimension prompting has targeted utility
- **Evidence:** Sharpness +0.036, color fidelity +0.015 SRCC (Gemini); sharpness +0.019, color fidelity +0.037 (GPT-4.1).
- **Tradeoff:** Overall quality degrades (-0.017 Gemini, -0.051 GPT-4.1). 3x latency and cost.
- **Confidence:** Moderate. n=44 is better than n=23 but still small; effect sizes are within plausible regression-to-mean range.

### F3: Few-shot examples are model-dependent
- **Evidence:** Few-shot hurt Gemini (-0.073 wSRCC) but helped Qwen (+0.097 wSRCC) at n=23.
- **Confidence:** Low-moderate. Large effect size for Gemini is encouraging, but n=23 caveats apply.

### F4: Multi-sample averaging is cost-prohibitive
- **Evidence:** 3x median at temp=0.3 gained +0.019 wSRCC (Gemini) at 12.5x latency.
- **Confidence:** Moderate. The cost-benefit ratio is clearly unfavorable regardless of exact effect size.

---

## Open Questions

### Q1: What is the minimum sample size for reliable prompt optimization?
- **Current answer:** n >= 200 (heuristic based on SE of SRCC).
- **Needed:** Formal power analysis. Simulate prompt optimization experiments at varying n (50, 100, 200, 500) with known ground truth to determine the sample size at which false-positive arm selection drops below 5%.
- **Priority:** High. This is the paper's central methodological claim.

### Q2: Does per-dimension prompting replicate at full scale?
- **Current data:** n=44 only. The A/B test was not validated at n=1000.
- **Needed:** Run 3-prompt evaluation on full DIQA-5000 test set for Gemini 3 Flash.
- **Estimated cost:** ~$9 (1000 images x 3 calls).
- **Priority:** Medium. The effect sizes are small enough that the answer may be "no meaningful difference."

### Q3: Does prompt strategy interact with document type?
- **Hypothesis:** Per-dimension prompting may help more on documents where quality dimensions are decorrelated (e.g., sharp text with poor color fidelity).
- **Needed:** Stratified analysis of A/B test data by document quality profile. Extend to synthetic OOD categories.
- **Priority:** Medium.

### Q4: Temperature-prompt interaction
- **Current design:** All experiments used temperature = 0.0.
- **Hypothesis:** Multi-sample averaging might be more effective at higher temperatures (more diverse samples to aggregate).
- **Needed:** Factorial experiment: {temp=0.0, 0.3, 0.7} x {1-sample, 3-sample median, 5-sample mean}.
- **Priority:** Low. Cost of multi-sample is prohibitive regardless.

### Q5: Soft-prompt tuning for open-weight models
- **Opportunity:** Qwen3-VL-8B is open-weight and could benefit from learned soft prompts.
- **Needed:** Fine-tune soft prompt tokens on DIQA-5000 training set, evaluate on test set.
- **Priority:** Low. Qwen3-VL-8B baseline performance is weak (wSRCC = 0.481), and soft-prompt gains are unlikely to close the gap to Gemini (0.708).

---

## Refinements to Current Analysis

### R1: Bootstrap power analysis for A/B test
- Run bootstrap hypothesis test on A/B data to determine whether per-dimension improvements are statistically significant at alpha = 0.05.
- Use paired bootstrap (same images in both conditions) for maximum power.

### R2: Per-image analysis of resize effect
- Identify which images benefit most from native resolution vs 1024px.
- Correlate with image properties (original resolution, aspect ratio, content type).
- Goal: understand *why* some images improve and others degrade, informing adaptive preprocessing.

### R3: Cost-efficiency frontier
- Plot wSRCC vs total cost for all prompt variants.
- Include multi-model consensus (from Paper 1) as a comparison point.
- Determine whether any prompt variant dominates the cost-efficiency frontier.

### R4: Qwen 3.5 Flash full-scale validation
- The Qwen optimization results exist at n=23 only. Full-scale validation would confirm or refute the model-specific interaction with few-shot examples.
- Estimated cost: ~$1 (Qwen pricing is 15x cheaper than Gemini).

---

## Future Experiments

### E1: Adaptive prompting
- **Design:** Two-stage pipeline. Stage 1: coarse single-prompt assessment. Stage 2: if confidence is low or dimensions disagree, re-evaluate with per-dimension prompts.
- **Expected benefit:** Capture per-dimension gains on difficult images without 3x cost on easy images.
- **Estimated cost reduction:** 50-70% vs uniform 3-prompt.

### E2: Rubric calibration study
- **Design:** Vary the scale anchors in the prompt (e.g., provide example scores for reference images, adjust the verbal descriptions of each quality level).
- **Goal:** Reduce the systematic over-rating bias (+0.5 to +1.5 MOS) documented in Paper 1.
- **Risk:** May improve calibration (MAE) at the expense of correlation (SRCC), which is the metric that matters for pseudo-labeling.

### E3: Chain-of-thought structured prompting
- **Design:** Require the model to assess specific quality attributes (blur severity, moire presence, color cast direction) before assigning a numeric score.
- **Hypothesis:** Structured reasoning forces attention to quality-relevant features, improving correlation.
- **Counter-evidence:** Qwen3-VL-8B Thinking (which uses CoT) performed *worse* than Instruct (wSRCC 0.409 vs 0.481), suggesting CoT may introduce overthinking artifacts.

### E4: Minimum viable prompt study
- **Design:** Systematically ablate prompt components (system prompt, scale anchors, output format instructions, reasoning request) to identify the minimal prompt that preserves baseline correlation.
- **Goal:** Reduce prompt token count and latency without sacrificing quality.

---

## Cross-Paper Dependencies

| Dependency | Direction | Description |
|-----------|-----------|-------------|
| Paper 1 (VLM Benchmark) | Input | Baseline wSRCC values, model rankings |
| Paper 2 (Cross-Domain) | Input | OOD category analysis for stratified prompt study |
| Paper 4 (OOD Detection) | Output | Validated baseline prompt used in OOD evaluation |
| Paper 7 (Pseudo-Labeling) | Output | Prompt configuration recommendation feeds into pipeline design |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-08 | Recommend single-prompt baseline at 1024px | Full-scale validation showed no-resize does not replicate; baseline is simplest and most robust |
| 2026-03-08 | Set minimum n=200 for prompt optimization | Based on SE analysis; formal power analysis (Q1) may refine this |
| 2026-03-08 | Deprioritize multi-sample averaging | 12.5x cost for +0.019 wSRCC is impractical at any scale |
| 2026-03-08 | Flag A/B test for full-scale validation | n=44 results are directionally interesting but not confirmatory |

---

## Peer Review Feedback Log

**Review date:** 2026-03-08
**Panel:** GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast (DeepSeek V3 unavailable)
**Consensus recommendation:** Minor Revision (3/4 Minor, 1/4 Major)
**Full review:** See `PEER_REVIEW.md`

### Action Items from Peer Review

#### High Priority (consensus across 3+ reviewers)

- [ ] **PR-1: Add sub-sampling power analysis.** Use existing n=1000 data to evaluate subsets at n={50, 100, 200, 500}. Plot SRCC variance and arm-ranking stability vs sample size. Validates the n >= 200 recommendation empirically. (Addresses Q1 and reviewer consensus.)
- [ ] **PR-2: Add prompt appendix.** Include all 7 prompt variants (system + user messages), JSON schema, and post-processing rules verbatim in the paper.
- [ ] **PR-3: Report paired bootstrap CI for deltas.** Compute bootstrap CI for the *difference* in wSRCC (baseline vs no-resize), not just overlapping CIs per condition.
- [ ] **PR-4: Consolidate Tables 2 and 4.** Merge into single comprehensive table to eliminate redundancy.

#### Medium Priority (raised by 2 reviewers)

- [ ] **PR-5: Expand full-scale validation.** Run at least one additional arm at n=1000 (2048px or few-shot) and/or validate Qwen baseline at n=1000. (~$1-3 cost.)
- [ ] **PR-6: Deepen few-shot interaction analysis.** The Gemini/Qwen divergence (-0.073 vs +0.097) needs mechanistic hypothesis and, if possible, additional testing.
- [ ] **PR-7: Acknowledge A/B test power limitations.** Either expand n=44 or explicitly state the limited statistical power for the observed effect sizes.

#### Low Priority / Minor Fixes

- [ ] **PR-8: Clarify preprocessing details.** Document resize algorithm, color space, compression format, API server-side processing.
- [ ] **PR-9: Fix Zhang et al. (2024) citation.** Either tie "Benchmark Data Contamination" explicitly to evaluation methodology or replace with a more relevant reference.
- [ ] **PR-10: Add SE derivation.** Cite or derive the "+/- 0.15 at 95% CI" claim for n=23 SRCC (Section 2.3).
- [ ] **PR-11: Specify exact model API strings.** Replace "GPT-4.1 (OpenAI)" with the exact API identifier used.
- [ ] **PR-12: Revise "regression to the mean" framing.** Gemini reviewer argues "small-sample variance/selection bias" is more precise. Consider refining Section 2.3 terminology.

### Reviewer Score Summary

| Criterion           | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | Mean |
| ------------------- | ------- | -------------- | --------- | ------------- | ---- |
| Technical Soundness | 3 | 4 | 4 | 4 | 3.75 |
| Completeness | 3 | 3 | 3 | 3 | 3.00 |
| Clarity | 4 | 5 | 4 | 5 | 4.50 |
| Novelty | 3 | 3 | 3 | 3 | 3.00 |
| Reproducibility | 3 | 5 | 4 | 4 | 4.00 |

---
