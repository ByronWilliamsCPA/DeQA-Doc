# Research Agenda: Paper 1 -- VLM Benchmark for Document Image Quality Assessment

**Status:** Living document
**Last updated:** March 2026
**Paper:** DeQA-Doc Technical Report 1/7

---

## Potential Improvements

- **Expand model coverage to 12+ VLMs.** The current benchmark evaluates 7 models, but several strong candidates were excluded: Gemini 2.0 Flash, GPT-4o, Claude Sonnet 4, Llama 4 Scout, Mistral-Large-Pixtral. Adding these would strengthen the generalizability of findings (particularly the over-rating pattern) and may identify models with better calibration. Expected impact: moderate -- unlikely to change top-1 ranking but may reveal better calibration properties. Effort: 2-3 days (API cost ~$150).

- **Per-model calibration functions with held-out validation.** Currently, calibration is demonstrated only on SigLIP2 predictions. Running all 7 VLMs on the 3,500 training images would enable learning per-model, per-dimension isotonic regression functions and evaluating calibrated wSRCC on the test set. Expected impact: high -- this directly enables the pseudo-labeling pipeline. Effort: 3-5 days (API cost ~$200, plus analysis).

- **Confidence interval methodology.** The current bootstrap CIs use naive percentile intervals with 1,000 iterations. Switching to BCa (bias-corrected and accelerated) intervals with 10,000 iterations would provide tighter, more accurate CIs, potentially resolving the overlapping intervals between Gemini 3 Flash and GPT-4.1. Expected impact: low on conclusions, moderate on statistical rigor. Effort: 0.5 days.

- **Inter-annotator agreement analysis.** DIQA-5000 provides only aggregate MOS, not per-annotator scores. If per-annotator data becomes available, comparing VLM-human agreement against human-human agreement would contextualize VLM performance relative to inherent task ambiguity. Expected impact: high for interpretation. Effort: depends on data availability.

- **Structured error analysis by document type.** The DIQA-5000 images span diverse content types (text, tables, diagrams, handwritten notes). Categorizing images by content type and analyzing VLM performance per category would identify whether VLMs struggle more with specific document structures. Expected impact: moderate. Effort: 2-3 days (requires image annotation or automated classification).

## Test Refinements

- **Increase prompt optimization sample size to n=200.** The current finding that small-sample optimization is unreliable is itself based on a comparison between n=23 and n=1,000. A dedicated n=200 experiment would empirically validate the recommended minimum sample size and establish whether the 7-arm rankings stabilize at this scale. This matters because n=200 is the actionable recommendation in the paper.

- **Test-retest reliability.** Run the same model (e.g., Gemini 3 Flash, temp=0) on the full test set twice, separated by 1 week, to measure temporal stability of VLM predictions. Any model update or routing change at the provider level would introduce noise. This directly affects whether VLM pseudo-labels are reproducible.

- **Stratified bootstrap by quality bucket.** Current CIs use uniform bootstrap sampling. Since the test set is heavily imbalanced (613 fair vs. 5 excellent), CIs may be dominated by the fair bucket. Stratified bootstrap (preserving bucket proportions in each resample) would provide more representative uncertainty estimates for the tails.

- **Add RMSE and Kendall's tau.** The paper reports SRCC, PLCC, MAE, and RMSE. Adding Kendall's tau-b would provide an alternative rank correlation metric less sensitive to outliers. Some IQA benchmarks report it, improving cross-paper comparability.

- **Evaluate parse failure handling.** Gemini 2.5 Pro had 70 parse failures (7.0%). Currently, metrics exclude these images. An alternative approach: assign the dataset median score to parse failures and recompute. This "worst-case" metric would better reflect production reliability.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|---|---|---|---|
| Multi-sample variance estimation | Running each VLM 3-5x with temp>0 provides variance estimates that correlate with human annotation disagreement | 3-5x API calls for top 3 models on 1,000 images | High |
| ~~Consensus scoring (2-3 models)~~ | ~~Averaging Gemini 3 Flash + GPT-4.1 predictions yields wSRCC > 0.72 on DIQA-5000~~ | ~~Already have all data; analysis only~~ | **Done** — Pair wSRCC = 0.744; All-7 weighted = 0.760 calibrated. See `research/consensus/RESULTS.md` |
| Temperature sweep | There exists an optimal temperature (0.1-0.5) that improves SRCC by reducing response stereotypy | 5 temperature values x 1,000 images x 2 models | Medium |
| Reference-anchored prompting | Providing a "typical fair quality" reference image in the prompt reduces over-rating bias by >0.3 MOS | 1,000 images x 2 models with 2-3 reference images | Medium |
| Chain-of-thought structured output | Asking models to first describe defects, then rate, improves SRCC for lower-ranked models | 1,000 images x 3 models with 2 prompt variants | Medium |
| Real-world OOD documents | VLM rankings on real-world OOD (receipts, historical manuscripts, handwritten forms) match synthetic OOD patterns | Need to source 200-500 real OOD documents with annotations | High |
| Cross-dataset transfer | VLM pseudo-labels generated on DIQA-5000-like documents transfer to Tobacco800 or RVL-CDIP quality assessment | Need quality annotations for alternative datasets | Low |
| Newer models (Gemini 2.5 Flash, GPT-4o-mini, Claude Sonnet 4) | Each new model generation improves wSRCC by 0.02-0.05 over predecessors | 1,000 images per model | Ongoing |
| Image resolution sweep (512, 768, 1024, 1536, 2048, native) | There is no monotonic relationship between resolution and SRCC; 1024 is near-optimal | 6 resolution variants x 1,000 images x 2 models | Low |
| Dimension-specific prompts at scale | Separate prompts improve color fidelity SRCC by >0.03 on full 1,000-image set | 3,000 additional API calls per model | Medium |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|---|---|---|---|---|
| Mar 2026 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Methodology | Add parse-failure sensitivity analysis (worst-case imputation + intersection set) | Open |
| Mar 2026 | 4-model consensus | Metrics | Define MainScore explicitly in Section 3.3 | Open |
| Mar 2026 | 4-model consensus | Reproducibility | Include verbatim prompts (system + user) in appendix | Open |
| Mar 2026 | 4-model consensus | Consistency | Fix category count mismatch: abstract says 13, Section 3.1 says 15 | Open |
| Mar 2026 | 4-model consensus | Consistency | Standardize model naming throughout (e.g., "Gemini 3 Flash Preview" vs "Gemini 3 Flash") | Open |
| Mar 2026 | 4-model consensus | Completeness | Add calibration scatterplot (raw vs. calibrated VLM predictions) | Open |
| Mar 2026 | 4-model consensus | Analysis | Analyze CoT failure traces to explain reasoning model underperformance | Open |
| Mar 2026 | 4-model consensus | Methodology | Specify PLCC 4-parameter logistic fitting procedure details | Open |
| Mar 2026 | 4-model consensus | Completeness | Add variance estimation pilot (multi-sample on subset) | Open |
| Mar 2026 | 4-model consensus | Framing | Reframe synthetic OOD claims as "agreement with degradation parameters" or validate with human ratings | Open |

---

## Cross-Paper Dependencies

| Downstream Paper | Dependency on Paper 1 | Status |
|---|---|---|
| Paper 2 (Cross-Domain) | Synthetic OOD results from Section 4.5 | Data complete |
| Paper 3 (Baseline vs Trained) | VLM benchmark results from Table 1 | Data complete |
| Paper 4 (OOD Detection) | VLM failure modes informing detection thresholds | Findings documented |
| Paper 5 (NR-IQA Baselines) | Landscape comparison from Appendix B | Data complete |
| Paper 7 (Pseudo-Labeling) | All VLM metrics, calibration, consensus design | Metrics complete; test-set calibration complete (see `research/consensus/`); training-set calibration pending (Handoff 03) |

## Open Questions

1. **Is the over-rating problem fixable with better prompts, or is it intrinsic to VLM training?** The universality across 7 models from 4 providers suggests it is a fundamental property of web-trained multimodal models encountering intentionally degraded images. If so, calibration is the only path, not prompt engineering.

2. **Why do reasoning models underperform?** The Qwen3-VL-8B Thinking result (0.409 vs. Instruct 0.481) is counterintuitive. Is CoT reasoning introducing noise by over-analyzing quality attributes? Is the thinking budget wasted on aspects irrelevant to quality ranking? This deserves a focused investigation.

3. **What is the minimum VLM-human correlation needed for effective pseudo-labeling?** The pseudo-labeling pipeline assumes wSRCC > 0.7 is sufficient, but this threshold is based on intuition rather than empirical validation. Training student models on labels of varying quality (simulated by adding noise to human labels) would establish the empirical floor.

4. **Can VLMs provide useful soft-label distributions, not just point estimates?** The DeQA-Score training pipeline requires (mu, sigma) pairs. VLMs currently provide only mu. If multi-sample variance estimates correlate poorly with human disagreement, the pipeline may need to use pseudo-variance injection instead.
