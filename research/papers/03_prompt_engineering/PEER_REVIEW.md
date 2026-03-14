# Peer Review: Paper 3 — Prompt Engineering
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3-0324 was unavailable on OpenRouter during the review session. The consensus is based on the 4 remaining models. All 4 models reviewed the full paper independently with neutral stance.

## Consensus Recommendation: Minor Revision

Three of four reviewers recommend **Minor Revision**; one (GPT-5.2) recommends **Major Revision**. The majority view is that the paper makes a valuable methodological contribution that is publishable with targeted improvements. The dissenting Major Revision opinion stems from wanting broader full-scale validation (multiple arms, multiple models at n=1000), which the majority considers desirable but not blocking.

---

## Consolidated Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 4 | **3.75** |
| Completeness | 3 | 3 | 3 | 3 | **3.00** |
| Clarity | 4 | 5 | 4 | 5 | **4.50** |
| Novelty | 3 | 3 | 3 | 3 | **3.00** |
| Reproducibility | 3 | 5 | 4 | 4 | **4.00** |

---

## Points of Universal Agreement

### Strengths (all 4 reviewers agree)

1. **Compelling negative result.** The pilot-to-full-scale reversal (+0.042 wSRCC at n=23 reversing to -0.009 at n=1000) is a striking and undeniable demonstration of small-sample evaluation pitfalls. This is the paper's strongest contribution.

2. **Exceptional clarity and organization.** The narrative arc from misleading pilot to full-scale reality check is engaging and well-structured. Tables are clear with sensible baseline comparisons. Writing quality is consistently praised (4-5/5 across all reviewers).

3. **Actionable cost-efficiency analysis.** The cost/latency tradeoff analysis (single-call vs multi-call tiers, 3x multiplier for per-dimension prompting, $15 total experimental cost) provides directly usable guidance for practitioners.

4. **Good artifact transparency.** Exact hyperparameters (temp=0.0, max_tokens=1024), cost estimates, random seeds, and file paths for all checkpoints and raw data are documented.

### Weaknesses (all 4 reviewers agree)

1. **The n >= 200 recommendation lacks formal justification.** All reviewers flag that the minimum sample size recommendation is based on heuristic SE reasoning, not a formal power analysis. The paper should either derive this formally or demonstrate it empirically via sub-sampling.

2. **Missing prompt templates.** The actual prompt text used in each arm is not included in the paper, only referenced as archived files. All reviewers agree this is a reproducibility gap that must be addressed with an appendix.

3. **Completeness gap: limited full-scale validation.** Only one arm (no-resize) was validated at n=1000, and only for one model (Gemini). This limits the generality of conclusions, particularly for the cross-model claims.

4. **Tables 2 and 4 are redundant.** Both present the same full-scale comparison data in slightly different formats. Consolidation would improve readability.

---

## Points of Disagreement

### Severity of statistical gaps (Minor vs Major Revision)

- **GPT-5.2** considers the lack of paired bootstrap CIs for deltas (not just overlapping CIs), missing multiple comparison corrections in the 7-arm pilot, and limited full-scale validation as collectively requiring Major Revision.
- **Gemini, Qwen, Grok** view these as targeted improvements that can be addressed without fundamentally restructuring the paper, warranting only Minor Revision.
- **Resolution**: The majority view is more appropriate for a technical report. The statistical gaps are real but the paper's central narrative (small samples mislead) does not depend on precise delta CIs — the direction reversal itself is the evidence.

### "Regression to the mean" framing

- **Gemini 3.1 Pro** argues the term is slightly imprecise. The core phenomenon is better described as "small-sample variance" and "selection bias" rather than classical regression to the mean. The paper conflates favorable noise in the pilot sample with the statistical concept of regression toward a population mean.
- **Other reviewers** accept the framing without objection.
- **Resolution**: The critique has merit. Consider revising Section 2.3 to use more precise terminology, or explicitly distinguish between the classical statistical concept and the specific mechanism at play here.

### Reproducibility assessment

- **Gemini 3.1 Pro** scores reproducibility 5/5, citing excellent hyperparameter documentation and artifact paths.
- **GPT-5.2** scores 3/5, emphasizing that proprietary model versioning (Preview, Flash models change over time) and missing preprocessing specifics (resize algorithm, compression format, server-side downscaling) limit exact replication.
- **Resolution**: Both perspectives are valid. The paper is reproducible in spirit (methodology is clear) but exact numerical replication is inherently limited by proprietary API versioning. Adding a note acknowledging this limitation would address the concern.

---

## Specific Suggestions for Improvement

### High Priority (consensus across 3+ reviewers)

1. **Add formal power analysis or empirical sub-sampling curve.** Use existing n=1000 predictions to evaluate subsets at n={50, 100, 200, 500}. Plot SRCC variance and arm-ranking stability vs sample size. This would make the n >= 200 recommendation empirically grounded and highly citeable.

2. **Include full prompt templates as appendix.** All seven prompt variants (system + user messages), the JSON output schema, and any post-processing rules (clamping, rounding) should be included verbatim.

3. **Report paired bootstrap CIs for deltas.** For Table 2 (baseline vs no-resize), compute and report the bootstrap CI for the *difference* in wSRCC, not just overlapping CIs for each condition independently.

4. **Consolidate Tables 2 and 4.** Merge into a single comprehensive table with all metrics (SRCC, PLCC, MAE, bias, latency) to eliminate redundancy.

### Medium Priority (raised by 2 reviewers)

5. **Expand full-scale validation.** Run at least one additional arm (e.g., 2048px or few-shot) at n=1000, and ideally validate Qwen at n=1000 for at least the baseline, to support cross-model generality claims.

6. **Deepen few-shot analysis.** The Gemini/Qwen divergence on few-shot prompting (-0.073 vs +0.097) is interesting but under-analyzed. Hypothesize specific mechanisms (pre-training calibration differences, in-context learning capacity) and test if possible.

7. **Acknowledge A/B test limitations.** Either expand n=44 to n >= 100, or explicitly state that the A/B test has limited statistical power for the small effect sizes observed (0.01-0.04 SRCC).

### Low Priority (raised by 1 reviewer)

8. **Clarify preprocessing specifics.** Document exact resize algorithm (bicubic/lanczos), color space handling, compression format, and whether API providers apply additional server-side processing.

9. **Add temperature discussion.** Briefly justify why temperature=0.0 was chosen and whether this interacts with prompt variant rankings.

10. **Specify exact model API strings.** "GPT-4.1 (OpenAI)" is ambiguous in the literature. Include the exact model identifier used in API calls.

---

## Minor Issues

| Issue | Source | Location |
|-------|--------|----------|
| SE approximation ("+/- 0.15 at 95% confidence") needs citation or derivation | GPT-5.2, Qwen | Section 2.3, Line 50 |
| Table 1 latency column appears Gemini-only; unclear if Qwen latencies differ | GPT-5.2 | Table 1, Line 99 |
| Zhang et al. (2024) "Benchmark Data Contamination" citation seems off-topic | GPT-5.2, Qwen | References, Line 245 |
| Abstract could state the reversal finding more prominently upfront | Qwen | Abstract, Line 12-14 |
| Add CIs to abstract's key numbers for consistency | Qwen | Abstract, Line 14 |
| DIQA-5000 license/terms should be briefly mentioned | Qwen | Section 6, Line 236 |

---

## Overall Assessment

This paper makes a genuinely valuable contribution to the VLM evaluation methodology literature. The central finding — that small-sample prompt optimization can actively mislead practitioners — is well-demonstrated and practically important. The writing quality is consistently excellent, and the cost-efficiency analysis is directly actionable.

The primary gaps are in statistical rigor (power analysis, paired delta CIs) and completeness (limited full-scale validation, missing prompt templates). These are addressable with modest additional effort and do not require rethinking the paper's structure or conclusions.

**Recommended path to acceptance:**
1. Add sub-sampling power analysis (can be done from existing data, no new API calls needed)
2. Add prompt appendix (copy from archived files)
3. Add paired bootstrap CI for baseline vs no-resize delta
4. Consolidate Tables 2/4
5. Fix minor issues (citation, SE derivation, model identifiers)

Estimated effort: 1-2 days of analysis and writing.
