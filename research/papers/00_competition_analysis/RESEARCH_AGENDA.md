# Research Agenda: Paper 0 --- Competition Analysis

## Potential Improvements

- **Obtain per-dimension breakdowns.** The current analysis uses estimated DimScores for the heatmap (Figure 3) because the official competition report provides limited per-dimension data for individual teams. If team-specific SRCC/PLCC values per dimension become available from fact sheets or supplementary materials, the heatmap should be updated with exact values.
- **Add confidence intervals to leaderboard.** The MainScore differences across the top four teams (0.924--0.929) are small enough that statistical significance is uncertain at n=1,000. Requesting raw prediction arrays from teams or organizers would enable bootstrap CI computation.
- **Expand baseline coverage.** The organizer-evaluated baselines (DBCNN, HyperIQA, StairIQA, MUSIQ, TReS, RichIQA) represent a subset of available NR-IQA models. Adding results from CLIP-IQA+, LIQE, NIQE, BRISQUE, and other pyiqa models (as explored in Paper 5) would provide a more complete baseline landscape.
- **Incorporate post-competition results.** If teams publish extended versions of their solutions or new models are evaluated on DIQA-5000, the leaderboard and analysis should be updated.
- **Add qualitative failure analysis.** Include representative example images where top models disagree significantly, stratified by degradation type and document category.

## Test Refinements

- **Statistical significance testing.** Apply Williams' test or Steiger's test to compare dependent correlations between the top four MLLM teams. Report whether the championship margin (0.002 over 2nd place) is statistically significant.
- **Effect size analysis.** Compute Cohen's d or similar effect sizes for the MLLM vs CNN and MLLM vs Generative comparisons to quantify the practical significance of architecture family differences.
- **Score distribution analysis.** Examine whether team predictions exhibit different distributional characteristics (e.g., variance, skewness, kurtosis) that might explain performance differences beyond correlation metrics.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| Independent baseline replication | Organizer-reported baseline scores are reproducible with public model weights and standard evaluation code | DIQA-5000 test set, pyiqa library | High |
| Ensemble ablation | Removing the weakest model from top-4 ensembles does not significantly degrade MainScore | Team prediction arrays | High |
| Architecture family meta-analysis | MLLM dominance holds across other document quality benchmarks (SmartDoc-QA, DocIQ) | Alternative DIQA benchmarks | Medium |
| TTA sensitivity analysis | TTA gains vary by degradation type; blur-degraded images benefit more than shadow-degraded ones | Per-image degradation labels, TTA predictions | Medium |
| Label strategy comparison | Pseudo-variance with sigma=0.2*range is not optimal; grid search over sigma values yields higher MainScore | DIQA-5000 train/val splits | Medium |
| Resolution ablation | Progressively downsampling inputs degrades MLLM performance faster than CNN performance | DIQA-5000 test images at multiple resolutions | Low |
| Cross-competition transfer | Models trained on DIQA-5000 transfer to ISRGC-Q or FIQA challenges with minimal fine-tuning | VQualA 2025 multi-challenge datasets | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Statistical Rigor | Add bootstrap CIs or significance tests for top-4 MainScore differences (0.001-0.005 gaps at n=1000) | Open |
| 2026-03-08 | All 4 reviewers | Missing Artifact | Phantom Figure 3 reference at line 229 --- embed figure or remove callout | Open |
| 2026-03-08 | All 4 reviewers | Completeness | BIT ssvgg missing MainScore unexplained --- add reason (code verification failure?) | Open |
| 2026-03-08 | GPT-5.2, Grok 4.1 | References | Consolidate duplicate refs [4]/[11]/[12] (all cite DeQA-Doc Gao et al.) | Open |
| 2026-03-08 | GPT-5.2 | Disclosure | Clarify author relationship to DeQA-Doc team (analyst vs. competitor) | Open |
| 2026-03-08 | 3 of 4 reviewers | Language | Temper absolute/causal claims ("performance ceiling," "complete failure," "fundamentally benefits") | Open |
| 2026-03-08 | Qwen 3.5+, Grok 4.1 | Completeness | Add computational cost comparison table (parameters, latency, GPU requirements) | Open |
| 2026-03-08 | 3 of 4 reviewers | Data Quality | Source exact per-dimension SRCC/PLCC from team reports; label inferred values explicitly | Open |
| 2026-03-08 | Consensus | Overall | Recommendation: Minor Revision (3 Minor / 1 Major). Mean scores: TS 4.0, Comp 3.75, Clarity 4.25, Novelty 3.25, Repro 4.0 | Open |
