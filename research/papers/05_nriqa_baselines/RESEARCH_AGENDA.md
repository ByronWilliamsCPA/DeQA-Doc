# Research Agenda: Paper 5 --- NR-IQA Baselines

## Potential Improvements

- **Expand model coverage.** Only 5 of the many pyiqa models were evaluated. CLIP-IQA+, LIQE, NIQE, BRISQUE, and CNNIQA are readily available and would strengthen the benchmark. StairIQA was planned but unavailable in pyiqa; a manual implementation or alternative source should be investigated.
- **Add bootstrap confidence intervals.** The current results report point estimates for SRCC and PLCC. Adding 95% CIs via bootstrap resampling (n=1,000 iterations) would quantify measurement uncertainty, particularly important for models near the tier boundaries.
- **Per-document-type stratification.** DIQA-5000 contains printed, handwritten, form, and receipt documents. Breaking results down by document type would reveal whether pretrained NR-IQA models transfer better for specific categories (e.g., printed documents may have quality characteristics closer to natural images).
- **Feature-level analysis.** Extract intermediate feature representations from pretrained models and apply centered kernel alignment (CKA) or representational similarity analysis (RSA) to compare feature spaces between natural IQA and document IQA tasks.

## Test Refinements

- **Statistical significance testing.** Apply Williams' test or Steiger's test to compare dependent correlations between model pairs. Current rankings are based on point estimates, but differences of 0.01-0.03 in MainScore may not be statistically significant at n=1,000.
- **Calibration analysis.** Beyond correlation, measure calibration (e.g., expected calibration error) of pretrained model scores against DIQA MOS values. The score distribution analysis (Section 4.5) suggests severe miscalibration, but this should be quantified formally.
- **Resolution sensitivity.** Test whether preprocessing (resizing, padding) affects pretrained model performance. Some models (MUSIQ) handle arbitrary resolutions while others (DBCNN, HyperIQA) require fixed input sizes; the resizing strategy may introduce systematic bias.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| Linear probe on pretrained features | Frozen NR-IQA backbones + trained linear head achieve >0.7 MainScore, closing 50%+ of the gap vs full fine-tuning | DIQA-5000 train split (3,500 images) | High |
| CLIP-IQA+ and LIQE benchmark | Vision-language pretrained IQA models transfer better than pure CNN models due to text understanding | pyiqa library access | High |
| Per-category breakdown | Printed documents show higher pretrained SRCC than handwritten ones due to closer resemblance to natural images | DIQA-5000 category labels | Medium |
| Multi-model score fusion | Simple averaging of 3+ pretrained model scores improves over best single model | Existing per-image scores | Medium |
| Fine-tuning curve (data efficiency) | 500 DIQA images suffice to reach 80% of full fine-tuning performance | DIQA-5000 train subsets | Medium |
| Cross-benchmark validation | Pretrained models that score well on DIQA-5000 also score well on other document quality datasets | Alternative DIQA benchmarks (if available) | Low |
| Distortion-type attribution | Attribute pretrained model performance to specific quality dimensions using gradient-weighted class activation maps | Per-image model outputs + gradients | Low |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | All 4 models | Statistical rigor | Add bootstrap 95% CIs for MainScore and per-dimension SRCC/PLCC; pairwise significance tests (Williams' test) for adjacent-ranked models | TODO |
| 2026-03-08 | All 4 models | Completeness | Add qualitative failure visualization: grid figure showing documents with GT MOS, VLM predictions, and NR-IQA predictions | TODO |
| 2026-03-08 | All 4 models | Editorial | Fix Figure 3 / Table 4 cross-reference error on line 149 | TODO |
| 2026-03-08 | GPT-5.2, Gemini | Completeness | Provide full synthetic per-dimension table (SRCC/PLCC per dimension for all 5 models) to support Section 4.3 claims | TODO |
| 2026-03-08 | Gemini 3.1 Pro | Technical | Fix PLCC/SRCC linearity interpretation (line 107): matched values after logistic fitting means fit preserves rank ordering, not that raw output is linear | TODO |
| 2026-03-08 | Gemini 3.1 Pro | Editorial | Fix arXiv placeholder citation: `arXiv:2412.05XXX` on line 236 | TODO |
| 2026-03-08 | Qwen 3.5+, GPT-5.2 | Completeness | Include NIQE/BRISQUE handcrafted baselines rather than deferring to future work | TODO |
| 2026-03-08 | Qwen 3.5+ | Analysis | Explain why MUSIQ transfers worst despite ViT architecture (multi-scale tokenization mismatch hypothesis) | TODO |
| 2026-03-08 | GPT-5.2 | Reproducibility | Clarify preprocessing/resize details per model (input size, crop policy, grayscale handling) | TODO |
| 2026-03-08 | GPT-5.2 | Editorial | Soften "exactly the degradation types" (line 143) and qualify "first systematic measurement" (line 25) | TODO |
| 2026-03-08 | Gemini 3.1 Pro | Editorial | Remove or footnote StairIQA mention (line 76) — adds no actionable value | TODO |
