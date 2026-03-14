# Research Agenda: Paper 7 --- Pseudo-Labeling Pipeline

## Potential Improvements

- **End-to-end expansion cycle validation**: Execute a complete pipeline iteration -- pseudo-label a targeted OOD document set, retrain SigLIP2, re-fit OOD detector, and measure SRCC maintenance on DIQA-5000 plus OOD improvement. This is the single most important validation step. Without it, the pipeline remains theoretical. Expected impact: validates or invalidates the entire approach. Effort: high (requires VLM inference on training set for calibration, OOD document collection, student retraining, full evaluation).

- **VLM calibration on DIQA-5000 training split**: Run Gemini 3 Flash and GPT-4.1 on all 3,500 training images to fit per-model, per-dimension isotonic calibration functions. Currently only SigLIP2 calibration is validated. Expected impact: enables actual pseudo-label generation with calibrated scores. Effort: medium (~$10 API cost, ~2 hours inference, straightforward analysis).

- **Data-calibrated gating thresholds**: The current sigma-squared (0.64) and entropy (1.2) thresholds never trigger on real data. Evaluate whether data-calibrated thresholds (sigma-squared p75 = 0.072, entropy p75 = 0.625) improve pseudo-label quality by filtering uncertain SigLIP2 predictions. Expected impact: may activate 16.9% low-weight tier that is currently dead code. Effort: low (threshold change only, but requires end-to-end validation to measure impact).

- **Weighted ensemble scoring**: Replace simple dual-model averaging with weighted consensus based on per-dimension VLM strengths. Gemini leads on in-distribution and non-Latin scripts; GPT-4.1 leads on adversarial and CJK. Per-category weighting could improve consensus quality. Expected impact: moderate improvement on edge cases. Effort: medium (requires category detection or embedding-based routing).

- **Iterative drift monitoring**: Implement score distribution tracking across expansion cycles to detect confirmation bias or score drift. Compare pseudo-label MOS distributions from cycle N to cycle N+1 and flag systematic shifts. Expected impact: prevents silent pipeline degradation over multiple iterations. Effort: low (post-hoc analysis after each cycle).

- **Soft-label sigma calibration**: The DeQA-Score default sigma_pseudo = 0.8 was tuned for natural IQA datasets. DIQA-5000 has mean annotation std = 0.47. Calibrating sigma_pseudo to the document domain could improve soft-label distribution quality. Expected impact: tighter distributions for high-agreement samples. Effort: low (hyperparameter sweep during retraining).

## Test Refinements

- **Real-world OOD evaluation**: Test OOD detector and VLM teachers on naturally-occurring OOD documents from Tobacco800, RVL-CDIP, CORD, and handwritten form datasets. The 13-model consensus unanimously ranked this as the highest priority. Why it matters: synthetic AUROC likely overestimates real-world detection performance. Data: public datasets available, but ground-truth quality annotations may need human collection for a small calibration set.

- **Calibration generalization test**: Fit isotonic calibration on DIQA-5000 and evaluate on a held-out document type (e.g., forms, receipts). If calibration functions do not transfer, per-domain calibration may be needed, requiring a small labeled seed set per target domain. Why it matters: calibration transferability is a core assumption of the pipeline. Data: requires VLM predictions on both DIQA-5000 and a separate document dataset with quality annotations.

- **Tier 2 VLM cross-validation effectiveness**: Measure whether Tier 2 validation (Qwen3-VL-8B cross-check for flagged images) actually improves pseudo-label quality. Currently the Tier 2 pathway is designed but not empirically validated for its impact on downstream student performance. Why it matters: if Tier 2 does not improve labels, it adds latency and cost without benefit. Data: requires running Tier 2 on the 54 flagged test images and comparing against human annotations.

- **Multi-iteration stability test**: Run 3+ expansion cycles on a controlled subset to measure whether OOD boundary contraction follows the expected pattern and whether in-distribution SRCC is maintained. Why it matters: the iterative design is the core theoretical contribution, but convergence properties are unknown. Data: requires multiple rounds of pseudo-labeling, retraining, and re-evaluation.

- **Per-dimension gating thresholds**: The current gating is dimension-agnostic. Sharpness has the highest Tier 2 veto rates across models (from threshold sensitivity analysis). Per-dimension thresholds may improve precision. Why it matters: different quality dimensions may have different reliability patterns. Data: threshold sensitivity data already collected.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| First expansion cycle (end-to-end) | SigLIP2 maintains SRCC > 0.90 on DIQA-5000 after retraining on pseudo-labeled OOD documents | 500-1000 OOD documents + VLM pseudo-labels + DIQA-5000 training set | Critical |
| Real-world OOD detector validation | Mahalanobis AUROC > 0.95 on naturally-occurring OOD documents (not just synthetic) | Tobacco800, RVL-CDIP, CORD embeddings | Critical |
| VLM calibration fitting | Per-model isotonic calibration reduces VLM MAE by >5x on DIQA-5000 training split | 3,500 training images rated by Gemini 3 Flash + GPT-4.1 | High |
| ~~Consensus vs. single-model pseudo-labels~~ | ~~Student trained on 2-model consensus labels outperforms single-model labels by >0.01 wSRCC~~ | ~~Same OOD documents labeled by 1 vs. 2 VLMs~~ | **Done (teacher-side)** — All-7 weighted 0.760 vs best single 0.694. Student retraining pending. See `research/consensus/RESULTS.md` |
| Active learning sampling | Inter-model disagreement identifies the most informative images for human annotation (annotation efficiency >3x random) | VLM predictions for candidate OOD documents + human annotations for disagreement subset | High |
| Binarized document preprocessing | Inverting binarized images before VLM scoring corrects negative SRCC to positive | 30 synthetic binarized documents + inverted versions | Medium |
| Per-class OOD modeling (GMM) | GMM with per-document-type components improves AUROC on real-world OOD vs. single Gaussian | Embeddings labeled by document type (script, layout, degradation) | Medium |
| Cross-dataset transfer | Pseudo-labels generated on DIQA-like documents transfer to improve Tobacco800 quality prediction | DIQA-5000 pseudo-labels + Tobacco800 quality annotations | Medium |
| Alternative OOD methods (energy/ODIN) | Energy-based or ODIN detection ensemble improves AUROC by 0.5-1% over Mahalanobis alone | Same embeddings, different scoring functions | Low |
| Conformal prediction for OOD | Conformal p-values provide calibrated uncertainty for OOD decisions, improving threshold selection | Same data, conformal calibration split | Low |
| Multi-generation iterative convergence | 5+ expansion cycles show monotonic OOD boundary contraction without DIQA-5000 regression | Multiple rounds of the full pipeline | Low (long-term) |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Critical | No end-to-end validation cycle executed --- pipeline central claim is unproven (unanimous, 4/4) | Open |
| 2026-03-08 | 4-model consensus | Critical | VLM calibration demonstrated on SigLIP2 not VLM teachers --- must fit per-model isotonic calibration on 3,500 training images (unanimous, 4/4) | Open |
| 2026-03-08 | 4-model consensus | Critical | Synthetic-only OOD evaluation --- must validate on real-world datasets (Tobacco800, RVL-CDIP, CORD) (unanimous, 4/4) | Open |
| 2026-03-08 | 4-model consensus | Important | "14x MAE reduction" headline misleading --- primarily scale mismatch [0,1] vs [1,5], not calibration difficulty (3/4) | **Addressed** — Paper §3.4 now reports actual VLM bias (+0.57 to +1.50 MOS) and calibration MAE reduction (2-4x via 5-fold CV linear). 14x framing replaced with empirical numbers. |
| 2026-03-08 | 4-model consensus | Important | Missing pipeline ablation study --- OOD gate on/off, single vs dual teacher, weighting schemes (3/4) | **Partially addressed** — Paper §3.3 now includes consensus ablation table (single → pair → top-3 → all-7, mean vs median vs weighted). OOD gate ablation still pending end-to-end cycle. |
| 2026-03-08 | 4-model consensus | Important | Dead code thresholds: sigma-sq (0.64) and entropy (1.2) never trigger on real data (3/4) | Open |
| 2026-03-08 | Grok 4.1 Fast | Minor | Inconsistent model naming: "Gemini 3 Flash Preview" vs "Gemini 3 Flash" | Open |
| 2026-03-08 | Gemini 3.1 Pro | Minor | Section 6.3 repeated "1." numbering in medium-term and longer-term lists | Open |
| 2026-03-08 | GPT-5.2 | Minor | MainScore vs wSRCC metric confusion across tables --- define formula at first use | Open |
| 2026-03-08 | Grok 4.1 Fast | Minor | Appendix B table headers ambiguous: "d_M OOD" should be "d_M Tier2" | Open |
| 2026-03-08 | GPT-5.2 | Reproducibility | Missing exact prompts/templates, parsing rules, model version strings, synthetic OOD generation recipe | Open |
| 2026-03-08 | Gemini 3.1 Pro | Suggestion | Add small calibration generalization test: human-labeled OOD subset with ID-fitted calibration curve | Open |
| 2026-03-08 | 4-model consensus | Overall | Consensus recommendation: Major Revision (mean score 3.85/5). See PEER_REVIEW.md for full synthesis | -- |
