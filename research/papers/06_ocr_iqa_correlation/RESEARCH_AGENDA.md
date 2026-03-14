# Research Agenda: Paper 6 --- OCR-IQA Correlation

## Potential Improvements

- **Document-specific MOS calibration**: The current DeQA MOS spans only 0.412 points across tiers (2.942-3.354 on a 1-5 scale). A monotonic mapping calibrated on the CER-MOS curve (e.g., isotonic regression) could widen the effective scale and improve thresholding for quality gating. Expected impact: sharper practical thresholds, improved precision-recall for quality gating. Effort: low (post-hoc calibration, no retraining).

- **Per-dimension quality-CER analysis**: The current study uses overall MOS only. Sharpness and color fidelity MOS are available in the DeQA-Doc output. Analyzing which quality dimension best predicts CER could reveal whether blur-specific or noise-specific scores are more actionable for OCR workflows. Expected impact: finer-grained quality gating. Effort: low (data already collected, analysis only).

- **Bootstrap confidence intervals for all metrics**: The correlation report currently stores point estimates. Adding bootstrap 95% CIs for SRCC and PLCC would strengthen statistical claims and allow comparison of correlation differences across engines. Expected impact: stronger statistical reporting. Effort: low (add to analysis script).

- **Non-linear regression models**: The PLCC values (0.388-0.553) are consistently lower than SRCC values (0.435-0.683), suggesting the MOS-CER relationship is monotonic but non-linear. Fitting logistic or polynomial regression and reporting the fit could better characterize the functional form. Expected impact: improved predictive models for CER from MOS. Effort: low-medium.

- **Threshold optimization curve**: For each engine, compute precision-recall curves for a binary "high CER" classifier using MOS thresholds. This would directly demonstrate the quality gating use case with concrete operating points. Expected impact: immediately actionable for production deployment. Effort: low.

## Test Refinements

- **Extend to WER and field-level accuracy**: CER measures character-level accuracy but does not capture word boundaries or structured field extraction. Adding WER (already partially collected) and field-level metrics (for FUNSD which has labeled fields) would test whether quality predicts different error types differently. Why it matters: production systems often care about field extraction, not raw characters. Data: WER is in dataset.jsonl; field-level would require re-running evaluation with FUNSD annotations.

- **Increase base image count**: 200 base images may be insufficient for robust per-engine CER distribution estimation, especially at tail tiers. Scaling to 500-1000 base images from the full FUNSD+ corpus would tighten confidence intervals. Why it matters: reduces variance in tier-level statistics. Data: FUNSD+ has 1,026 training images available. Compute: proportional increase in OCR and DeQA inference (~3-5x). Note: now 9 engines (up from 4), so compute cost per base image has increased.

- **Expand VLM OCR coverage**: Current VLM OCR engines (GLM-OCR 0.5B, DeepSeek-OCR2 3B) are small models. Testing larger VLM-based OCR (GPT-4o, Gemini) would determine whether the architectural divide persists at scale or is a small-model artifact. Expected impact: validate or refine the architectural divide finding. Effort: medium (API costs for 1,200 images).

- **Add real-world degradation test set**: Collect 50-100 naturally degraded documents (camera captures, photocopies, fax transmissions) with known OCR ground truth. Running the same analysis on non-synthetic degradations would validate external generalizability. Why it matters: synthetic-only results may overestimate correlation strength. Data: would require new data collection and annotation.

- **Cross-validation of correlation estimates**: Split base images into 5 folds and compute per-fold SRCC estimates. This would provide variance estimates for the correlation coefficient itself, beyond bootstrap CIs which only resample within the full dataset. Why it matters: verifies stability of correlation estimates across document subpopulations. Data: no new data needed.

- **Multi-script and multi-language extension**: Add Arabic (right-to-left), Chinese (CJK), and mixed-script documents to test whether quality-CER correlations hold across script families. OCR engines may show different quality sensitivities for different scripts. Data: requires annotated document corpora with ground truth for each script.

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| ~~VLM zero-shot quality-CER correlation~~ | ~~Partially addressed: GLM-OCR and DeepSeek-OCR2 tested as VLM OCR engines. Result: VLM OCR engines show weak quality-CER correlation (SRCC -0.339 to -0.343), an architectural divide from traditional engines. Frontier VLMs as quality *predictors* remain untested.~~ | ~~Completed for OCR; quality prediction still open~~ | ~~Done (OCR) / High (quality)~~ |
| Quality-adaptive OCR routing | MOS-based engine routing reduces mean CER by >10% vs. single-engine baseline at equivalent cost | Same dataset + cost model per engine | High |
| Image enhancement before OCR | Applying denoising/super-resolution to LOW/DEGRADED images improves CER more than routing to robust engine | Same dataset + enhancement pipeline (Real-ESRGAN, NAFNet) | Medium |
| Qwen2.5-VL DeQA variant correlation | The Qwen2.5-VL-based DeQA model produces stronger CER correlations than mPLUG-Owl2 due to better document understanding | Same dataset + Qwen2.5-VL inference | Medium |
| Transfer to SROIE/CORD benchmarks | Quality-CER correlations generalize to receipt/invoice documents with SRCC > 0.4 | SROIE or CORD datasets with GT text, same distortion pipeline | Medium |
| Distortion-type decomposition | Blur-specific distortions produce stronger quality-CER correlation than noise or compression | Modified distortion pipeline (single-distortion tiers) | Low |
| Temporal degradation simulation | Simulated multi-generation photocopying produces monotonically increasing CER tracked by MOS | Custom pipeline with iterative scan-print-scan simulation | Low |
| OCR confidence vs. MOS correlation | OCR engine confidence scores correlate with MOS, enabling confidence-weighted quality fusion | Engine confidence extraction (available for Tesseract, Google Vision) | Medium |

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| 2026-03-08 | 4-model consensus (GPT-5.2, Gemini 3.1 Pro, Qwen 3.5+, Grok 4.1 Fast) | Statistical Rigor | Add cluster-bootstrapped CIs for Table 1 (unpaired) correlations to address non-independence of 6 variants per base document. Paired analysis (Table 2) already controls for this. | Open |
| 2026-03-08 | 4-model consensus | Baselines | Add BRISQUE and/or NIQE baseline comparisons on the same 1,200 images to establish whether DeQA-Doc offers incremental predictive value over traditional NR-IQA. | Open |
| 2026-03-08 | 4-model consensus | Data Claim | Qualify monotonicity claim in Section 4.3 --- LOW/DEGRADED tiers show non-monotonic CER and MOS, consistent with catastrophic failure plateau but contradicts "monotonically" wording. | Open |
| 2026-03-08 | 4-model consensus | Scale Calibration | Propose lightweight calibration mapping (isotonic regression or percentile stretching) to expand compressed MOS range (0.412 on 1-5 scale) for practical thresholding. | Open |
| 2026-03-08 | 4-model consensus | External Validity | Add born-digital PDF control set and/or naturally degraded documents to validate synthetic-only correlation results. | Open |
| 2026-03-08 | 4-model consensus | Metrics | Add WER alongside CER to capture word-boundary and segmentation failure modes. | Open |
| 2026-03-08 | 4-model consensus | Minor | Fix Figure 4 reference (no figures in manuscript), replace arXiv placeholder citation, clarify 0-1 vs 1-5 scale in tier table. | Open |
