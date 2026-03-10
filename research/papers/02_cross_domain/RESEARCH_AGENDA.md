# Research Agenda: Paper 2 — Cross-Domain Generalization

## Potential Improvements

- **Multi-model consensus scoring**: Combine GPT-4.1 and Gemini 3 Flash predictions (complementary strengths) into a weighted ensemble. Expected impact: 5-10% wSRCC gain on difficult OOD categories. Effort: Low (data already collected).

- **Category-specific confidence calibration**: Train per-category calibration models so that pseudo-label confidence reflects actual reliability for each OOD type. Expected impact: More accurate gating decisions in the pipeline. Effort: Medium.

- **Expand OOD taxonomy**: Current 13 categories miss important document types (handwritten, mixed media, degraded color photographs of documents). Expected impact: Better coverage of real-world distribution shifts. Effort: High (requires new data generation).

- **Fine-grained failure mode analysis**: Investigate why binarized and pristine categories fail universally — is it a prompt issue, a visual representation issue, or a fundamental limitation? Expected impact: Could unlock currently-failed categories. Effort: Medium.

## Test Refinements

- **Increase synthetic sample size**: Current n=40 per category (520 total) limits statistical power. Expand to n=100+ per category for tighter confidence intervals. Data: Synthetic generation pipeline. Compute: Minimal for generation, ~$50 for VLM re-evaluation.

- **Add real-world OOD samples**: Synthetic OOD may not capture all failure modes of real documents. Source real non-Latin script documents and extreme degradations from archival collections. Why it matters: Validates whether synthetic findings transfer to production.

- **Per-dimension OOD analysis**: Current analysis focuses on wSRCC aggregate. Break down per-dimension (overall, sharpness, color_fidelity) to identify dimension-specific failure modes. Data: Already available in checkpoints.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|---|---|---|---|
| Ensemble VLM pseudo-labeling | GPT-4.1 + Gemini 3 Flash ensemble outperforms either alone on OOD | Existing checkpoint data | High |
| Script-specific prompts | Tailored prompts for non-Latin scripts improve quality assessment | New VLM evaluations (~$20) | High |
| Fine-tuning on OOD data | Adding small OOD training set improves cross-domain generalization | 200+ annotated OOD images | Medium |
| Temporal stability test | VLM OOD performance is stable across API versions | Re-evaluation over 3 months | Medium |
| Active learning for OOD | Use OOD detector to prioritize human annotation of informative samples | OOD detector + annotation budget | Medium |
| Cross-model agreement as confidence | Agreement between VLMs correlates with prediction accuracy on OOD | Existing multi-model data | **Partially done** — Pairwise agreement heatmap in `research/consensus/figures/`. Full confidence-accuracy correlation still TODO |
| Degradation interpolation | Performance degrades smoothly between ID and OOD categories | Graded synthetic degradation data | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|---|---|---|---|---|
| 2026-03-08 | All 4 reviewers | Statistical rigor | Add bootstrap 95% CIs for all SRCC/wSRCC values (Tables 1, 2, A1) | TODO |
| 2026-03-08 | All 4 reviewers | Ground truth | Validate synthetic MOS with human annotations on ~50-100 stratified OOD subset (esp. Tier C) | TODO |
| 2026-03-08 | All 4 reviewers | Reproducibility | Include exact VLM prompt template as appendix | TODO |
| 2026-03-08 | GPT-5.2, Gemini | Parse failures | Audit parse failures by category; add sensitivity analysis (worst/best-case imputation) | TODO |
| 2026-03-08 | GPT-5.2, Gemini | Table clarity | Resolve Table 1 ID column mixing DIQA-5000 and synthetic ID subset metrics | TODO |
| 2026-03-08 | GPT-5.2, Grok | Experiments | Quantitatively test multi-model consensus (mean/median ensemble) on existing data | **Done** — Full consensus analysis in `research/consensus/`. Best pair OOD wSRCC = 0.778 (Gemini 3 Flash + GPT-4.1). All-7 weighted OOD wSRCC = 0.753. Paper updated §5.1-5.2. |
| 2026-03-08 | Gemini | Analysis | Explain Claude Haiku Tibetan (0.383) vs Myanmar (0.833) anomaly | TODO |
| 2026-03-08 | Qwen | Writing | Condense Sections 4.2-4.3; move per-model detail to appendix | TODO |
| 2026-03-08 | GPT-5.2 | Writing | Soften causal claims (e.g., "confirms" -> "consistent with") where synthetic evidence is suggestive | TODO |
| 2026-03-08 | Grok | Integration | Include OOD detection integration results from Paper 4 | TODO |
