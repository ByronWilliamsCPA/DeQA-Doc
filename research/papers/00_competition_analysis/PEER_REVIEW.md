# Peer Review: Paper 0 --- Competition Analysis
**Date**: March 2026
**Panel**: GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast, DeepSeek V3
**Format**: 5-model consensus peer review
---

> **Note**: DeepSeek V3-0324 was unavailable on OpenRouter during the review session. This consensus is based on 4 of 5 panelists. All other models completed their reviews successfully.

## Consensus Recommendation: Minor Revision

Three of four reviewers recommend **Minor Revision**; one (GPT-5.2) recommends **Major Revision**. The consensus recommendation is **Minor Revision** with attention to the statistical significance and provenance concerns raised unanimously.

## Aggregate Scores

| Criterion | GPT-5.2 | Gemini 3.1 Pro | Qwen 3.5+ | Grok 4.1 Fast | **Mean** |
|-----------|---------|----------------|-----------|---------------|----------|
| Technical Soundness | 3 | 4 | 4 | 5 | **4.00** |
| Completeness | 3 | 4 | 4 | 4 | **3.75** |
| Clarity | 4 | 4 | 4 | 5 | **4.25** |
| Novelty | 2 | 4 | 3 | 4 | **3.25** |
| Reproducibility | 3 | 5 | 3 | 5 | **4.00** |

## Unanimous Concerns (All 4 Reviewers)

### 1. No Statistical Significance Testing for Top-4 Score Gaps

All reviewers flagged that MainScore differences of 0.001--0.005 across the top four MLLM teams may not be statistically significant at n=1,000 test images. The paper draws ranking and architectural conclusions from these margins without bootstrap confidence intervals, Williams' test, or any uncertainty quantification.

**Action required**: Either add bootstrap CIs for MainScore differences or explicitly state that the top-4 scores represent statistical parity / an MLLM performance ceiling.

### 2. Phantom Figure 3 Reference

Line 229 references "Figure 3" for per-dimension estimates, but no figures are embedded in the manuscript text. This is a broken reference.

**Action required**: Either embed the referenced figure or remove the callout. If the figure exists in `generate_figures.py` output, include it inline.

### 3. BIT ssvgg Missing MainScore Unexplained

Team 7 (BIT ssvgg) lacks a MainScore in the summary table and text, but the paper claims to analyze "all seven verified submissions." No explanation is given for why the score is missing (e.g., code verification failure, late submission, withdrawal).

**Action required**: Add one sentence explaining the circumstance (e.g., "BIT ssvgg did not pass the mandatory code reproducibility verification").

### 4. Duplicate References

References [4], [11], and [12] all cite DeQA-Doc (Gao et al.) in different venues/formats. This is redundant.

**Action required**: Consolidate into a single canonical citation or clarify in a footnote why each is separately needed (e.g., preprint vs. camera-ready vs. supplementary).

## Majority Concerns (3+ Reviewers)

### 5. Claims Outpace Quantitative Evidence

Several high-level claims are stated more strongly than the data supports:
- "performance ceiling" (Section 5.1) --- not justified without significance testing
- "label engineering matters more than architecture" (Section 6.1) --- requires ablation or controlled comparison to prove
- "fundamentally benefits from visual-linguistic reasoning" (Section 5.1) --- alternative explanations (scale, pretraining data overlap, compute) not discussed
- "near-complete inability" / "complete failure" of DBCNN --- editorialized language

**Action required**: Temper causal/absolute language or provide supporting evidence. Add alternative hypotheses for MLLM dominance (e.g., model scale, pretraining corpus overlap).

### 6. Missing Per-Dimension Breakdowns

The paper acknowledges this limitation but could strengthen the analysis with per-dimension SRCC/PLCC tables for each team, sourced from team fact sheets or organizer reports.

**Action required**: Source exact per-dimension metrics from available team reports and add a comprehensive table. Label any inferred/estimated values explicitly.

### 7. Missing Computational Cost Analysis

No discussion of computational costs for MLLM deployment vs. CNN alternatives in production scenarios. Given the practical implications discussed in Section 6.1, this is a notable gap.

**Action required**: Add a table or paragraph comparing approximate inference costs (parameters, latency, GPU requirements) across architecture families.

## Individual Reviewer Highlights

### GPT-5.2 (Most Critical --- Major Revision)

**Unique contributions**:
- Flagged authorship/conflict-of-interest ambiguity: paper author is Byron Williams but Section 4.1 says "Team DeQA-Doc (led by Junjie Gao)." Relationship should be clarified.
- Suggested adding a "Data provenance" appendix mapping each team's details to source URL + section/page.
- Noted citation quality issues: [6] is incomplete (no year/venue/authors); [3] is non-archival.
- Recommended a sensitivity analysis showing how small SRCC/PLCC changes would affect MainScore ranking.

### Gemini 3.1 Pro (Minor Revision)

**Unique contributions**:
- Praised the "missing variance problem" taxonomy (pseudo-variance injection, linear interpolation, explicit Gaussian modeling) as an original strategic insight.
- Noted redundancy at line 232 (BIT ssvgg note repeats line 149).
- Highlighted that the paper successfully frames the transition from "data engineering over architecture" as the primary performance lever.

### Qwen 3.5+ (Minor Revision)

**Unique contributions**:
- Flagged formatting inconsistency: "0.924--0.929" uses double-hyphen inconsistently throughout.
- Noted path reference mismatch: `research/diqa_1.md` (line 250) does not match the repository structure shown at line 6.
- Suggested expanding production deployment cost analysis.
- Noted Abstract/Conclusion redundancy that could be tightened.

### Grok 4.1 Fast (Most Positive --- Minor Revision)

**Unique contributions**:
- Gave perfect scores for Technical Soundness (5/5), Clarity (5/5), and Reproducibility (5/5).
- Suggested independently verifying 2--3 top team codes (e.g., DeQA-Doc GitHub at ref [13]) and reporting verification results in Section 8.
- Recommended expanding Section 6.3 with cross-dataset experiments (e.g., DocIQ) for generalization claims.
- Noted that the "March 2026" date may confuse readers if the paper is preprinted early.

## Consensus Strengths (Cited by All/Most Reviewers)

1. **Exceptional thematic extraction**: The four cross-cutting themes (MLLM dominance, missing variance problem, native resolution, TTA) provide valuable synthesis not previously documented in the DIQA literature.

2. **Comprehensive coverage**: All 7 teams and 6 baselines are cataloged with architectural details, training strategies, and innovations in a well-structured format.

3. **Transparent limitations**: Section 6.3 honestly acknowledges missing significance tests, inferred estimates, and incomplete data --- a quality often missing in competition retrospectives.

4. **Practical deployment insights**: TTA as cost-free performance boost, ensemble diminishing returns, and label engineering emphasis provide actionable guidance beyond academic metrics.

## Consensus Weaknesses (Cited by All/Most Reviewers)

1. **Statistical rigor gap**: The absence of significance testing undermines the paper's ability to draw conclusions from narrow MainScore margins.

2. **Incomplete coverage**: BIT ssvgg's missing data and inferred per-dimension estimates weaken the "comprehensive" framing.

3. **Overstatement tendency**: Several claims use absolute or causal language ("complete failure," "fundamentally benefits," "performance ceiling") that the descriptive evidence does not fully support.

## Actionable Revision Checklist

- [ ] Add bootstrap CIs or significance tests for top-4 MainScore differences
- [ ] Embed Figure 3 or remove phantom reference
- [ ] Explain BIT ssvgg's missing MainScore
- [ ] Consolidate duplicate references [4]/[11]/[12]
- [ ] Temper absolute/causal language throughout
- [ ] Add authorship/role disclosure note (analyst vs. competitor)
- [ ] Complete incomplete citation [6]
- [ ] Add computational cost comparison table
- [ ] Source per-dimension metrics from team reports
- [ ] Fix path reference mismatch (line 250)
- [ ] Tighten Abstract/Conclusion redundancy
- [ ] Consistent dash formatting for score ranges

## Minor Issues Compilation

| Location | Issue | Reviewer |
|----------|-------|----------|
| Line 101 | "near-complete inability" --- overstatement | GPT-5.2 |
| Line 149/232 | BIT ssvgg note repeated redundantly | Gemini 3.1 Pro |
| Line 171 | "complete failure" --- overstatement | GPT-5.2 |
| Line 229 | Phantom Figure 3 reference | All |
| Line 250 | Path "research/diqa_1.md" mismatch | Qwen 3.5+ |
| Lines 274-291 | Duplicate references [4]/[11]/[12] | GPT-5.2, Grok 4.1 |
| Line 278 | Citation [6] incomplete (no year/venue) | GPT-5.2 |
| Line 4 | "March 2026" date may confuse | Grok 4.1 |
| Throughout | Double-hyphen "0.924--0.929" inconsistent | Qwen 3.5+ |

---

*Generated via multi-model consensus peer review. Each reviewer independently evaluated the paper without access to other reviewers' assessments.*
