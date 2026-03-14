# Peer Review: Paper 5 — NR-IQA Baselines
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3-0324 was unavailable on OpenRouter at review time. Four of five models provided reviews. All scores and recommendations below reflect the 4-model panel.

## Consensus Recommendation: Minor Revision

Three of four reviewers recommended **Minor Revision**; one (GPT-5.2) recommended **Major Revision**, primarily due to concerns about standalone reproducibility and missing statistical rigor. The consensus view is that the paper's core findings are sound and valuable, but several addressable gaps prevent immediate acceptance.

## Criterion Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | Mean | Consensus |
|-----------|---------|----------------|-----------|---------------|------|-----------|
| Technical Soundness | 4 | 4 | 4 | 5 | **4.25** | Methods are standard and valid (SRCC/PLCC with logistic fitting, MainScore). One technical flaw identified in PLCC/SRCC linearity interpretation. |
| Completeness | 3 | 4 | 3 | 4 | **3.50** | Core benchmark is solid for 5 models, but missing CIs, full synthetic per-dimension table, and handcrafted baselines (NIQE/BRISQUE). |
| Clarity | 4 | 4 | 4 | 5 | **4.25** | Exceptionally well-organized writing with effective tables. Marred by Figure 3/Table 4 cross-reference error and a few over-strong claims. |
| Novelty | 3 | 3 | 4 | 4 | **3.50** | Empirical consolidation rather than methodological novelty. The synthetic-vs-real transfer gap and three-tier hierarchy are valuable new contributions. |
| Reproducibility | 3 | 5 | 4 | 5 | **4.25** | Strong artifact documentation. Weakened by reliance on externally reported fine-tuned/VLM baselines and unspecified preprocessing details. |

**Overall: 3.95 / 5.00**

## Points of Agreement (Unanimous)

1. **Missing statistical uncertainty quantification** — All four reviewers flagged the absence of bootstrap 95% confidence intervals and pairwise significance tests (e.g., Williams' test) as the most important gap. Differences like 0.453 vs 0.437 MainScore cannot be declared meaningful without CIs at n=1,000.

2. **Figure 3 / Table 4 cross-reference error** — Line 149 states "Figure 3 shows the unified leaderboard" but immediately presents Table 4. No Figure 3 exists in the manuscript.

3. **Qualitative visual examples needed** — All reviewers requested a figure showing example document images with ground-truth MOS alongside NR-IQA and VLM predictions to visually demonstrate the domain gap and failure modes.

4. **Synthetic-vs-real gap is the strongest contribution** — The finding that pretrained models perform 10-77% better on synthetic documents than real ones, because synthetic degradations overlap with KonIQ-10K training distributions, was unanimously praised as a valuable and novel empirical insight.

5. **Three-tier performance hierarchy is useful framing** — The unified leaderboard contextualizing pretrained NR-IQA, zero-shot VLMs, and fine-tuned specialists was recognized as highly practical.

## Points of Disagreement

| Issue | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast |
|-------|---------|----------------|-----------|---------------|
| Overall recommendation | Major Revision | Minor Revision | Minor Revision | Minor Revision |
| Reproducibility score | 3/5 | 5/5 | 4/5 | 5/5 |
| Standalone completeness | Requires inline VLM protocol | External refs acceptable | Acceptable with caveats | Acceptable (script provided) |
| Technical soundness | 4/5 (wants preprocessing details) | 4/5 (PLCC linearity flaw) | 4/5 (wants MUSIQ analysis) | 5/5 (fully sound) |

GPT-5.2 was the strictest reviewer, emphasizing that the paper's reliance on externally reported baselines (competition fine-tuned scores, Paper 1 VLM scores) undermines standalone reproducibility. The other three reviewers considered this acceptable for a paper explicitly positioned within a technical report series.

## Top 3 Strengths (Consensus)

1. **Clear empirical quantification of the domain gap**: The paper provides the first systematic measurement of natural-to-document transfer for five widely-used NR-IQA models, using a competition-relevant metric. The finding that the best pretrained model achieves only 57% of its fine-tuned score is immediately actionable.

2. **Synthetic-vs-real transfer analysis**: The observation that synthetic degradations artificially inflate pretrained model performance (TReS: +77% relative improvement on synthetic vs real) protects future researchers from overestimating generic model capabilities based on synthetic-only testing.

3. **Unified cross-paradigm leaderboard**: Contextualizing pretrained NR-IQA against both zero-shot VLMs and fine-tuned specialists establishes a practical three-tier hierarchy that directly informs system design decisions for document quality pipelines.

## Top 3 Weaknesses (Consensus)

1. **No statistical uncertainty quantification**: Point-estimate rankings without confidence intervals or significance tests undermine claims about model ordering, especially in the 0.422-0.490 MainScore range where four models cluster tightly.

2. **Incomplete synthetic-set reporting**: The discussion of PLCC-vs-SRCC divergence on synthetic data (Section 4.3) references specific per-dimension values (e.g., HyperIQA PLCC_O=0.798 vs SRCC_O=0.639) without providing a full synthetic per-dimension table mirroring Table 1.

3. **Partially imported baselines**: Comparisons to fine-tuned models rely on competition-reported numbers, and VLM comparisons rely on Paper 1 results. This reduces standalone reproducibility and means the paper's strongest claims depend on external validation.

## Specific Suggestions for Improvement

### High Priority (Unanimous or Near-Unanimous)

1. **Add bootstrap 95% CIs** for MainScore and per-dimension SRCC/PLCC on both test splits. Report pairwise significance tests (Williams' test) for adjacent-ranked models. This is the single most impactful revision.

2. **Fix the Figure 3 / Table 4 cross-reference** (line 149). Either embed the actual Figure 3 from the figure generation script or update the text to reference Table 4.

3. **Add a qualitative failure visualization**: A grid figure showing 2-3 document images with ground-truth MOS, best VLM prediction, and best NR-IQA prediction to demonstrate what semantic document traits the NR-IQA models miss.

4. **Provide the full synthetic per-dimension table** mirroring Table 1, with SRCC and PLCC per dimension for all 5 models on the synthetic split. This is needed to support the PLCC-vs-SRCC divergence claims in Section 4.3.

### Medium Priority (Raised by 2+ Reviewers)

5. **Fix the PLCC/SRCC linearity interpretation** (line 107). The claim that "the prediction-to-MOS relationship is approximately linear" because SRCC and PLCC are closely matched is technically incorrect. Matched SRCC and PLCC after 4-parameter logistic fitting means the nonlinear fit tightly preserves rank ordering, not that raw model output scales linearly against MOS.

6. **Include NIQE and BRISQUE** in the benchmark rather than deferring to future work. These handcrafted baselines are trivially available in pyiqa and would strengthen the paper's coverage and claims about "off-the-shelf" model failure.

7. **Explain why MUSIQ transfers worst** despite the ViT architecture generally being considered stronger than CNNs for transfer learning. The 4.6x improvement gap suggests MUSIQ's multi-scale tokenization is specifically mismatched to document features, which deserves analysis.

8. **Clarify preprocessing details per model**: Input resize/crop policy, grayscale vs RGB handling, and whether any multi-crop evaluation is used. MUSIQ handles arbitrary resolutions while DBCNN/HyperIQA require fixed input sizes; the resizing strategy may introduce systematic bias.

### Low Priority (Single Reviewer)

9. **Soften "exactly" language** (line 143): "exactly the degradation types represented in KonIQ-10K" should be qualified unless distortion distributions are verified.

10. **Fix arXiv placeholder citation** (line 236): Reference 6 contains `arXiv:2412.05XXX`.

11. **Move or remove StairIQA mention** (line 76): "A sixth model (StairIQA) was planned but unavailable" adds no value; move to footnote or delete.

12. **Qualify "first systematic measurement" claim** (line 25) or back with additional citations confirming no prior DIQA transfer studies exist.

## Minor Issues

| Location | Issue | Reviewer(s) |
|----------|-------|-------------|
| Line 149 | "Figure 3 shows..." but content is Table 4 | All 4 |
| Line 236 | Unresolved arXiv ID: `arXiv:2412.05XXX` | Gemini |
| Line 143 | Over-strong: "exactly the degradation types" | GPT-5.2 |
| Line 76 | StairIQA mention adds no value | Gemini |
| Line 107 | PLCC/SRCC linearity claim is technically incorrect | Gemini |
| Line 25 | "first quantitative measurement" may be overstated | GPT-5.2 |

## Reviewer-Specific Unique Insights

**GPT-5.2** (strictest): Emphasized that the paper should be made standalone by either including the VLM evaluation protocol inline or explicitly marking VLM results as external. Suggested characterizing distortion types/frequencies in the synthetic set to strengthen causal claims.

**Gemini 3.1 Pro**: Identified the specific technical flaw in the PLCC/SRCC linearity argument. Gave perfect reproducibility score (5/5), noting that Section 6 provides exactly what is needed to replicate the NR-IQA portion of the study. Suggested the PLCC advantage on synthetic data "inflates MainScore relative to what SRCC alone would suggest."

**Qwen 3.5+**: Most generous on novelty (4/5), recognizing the 52% VLM advantage finding and synthetic-vs-real transfer pattern as useful contributions. Uniquely requested analysis of why MUSIQ transfers worst despite ViT architecture strengths.

**Grok 4.1 Fast**: Most positive overall (23/25 total). Noted that the figure generation script is provided even though figures are not embedded, partially mitigating the missing Figure 3 issue. Praised the paper as "exemplary" for reproducibility.

## Summary Assessment

The paper makes a clear, practical contribution: off-the-shelf NR-IQA models are inadequate for document quality assessment, zero-shot VLMs provide a better alternative, and domain-specific fine-tuning remains necessary for best results. The synthetic-vs-real gap analysis is a particularly valuable finding. The primary revisions needed are statistical (add CIs), completeness (full synthetic table, qualitative examples), and editorial (fix cross-references, soften over-strong claims). None of these require new experiments beyond what the existing data supports. With these revisions, the paper would be suitable for acceptance.
