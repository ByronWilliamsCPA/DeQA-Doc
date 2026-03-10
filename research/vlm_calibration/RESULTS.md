# VLM Teacher Calibration Results

## Summary

Per-model isotonic calibration fitted on DIQA-5000 training split, evaluated on 1,000 test + 520 OOD images.

**Training data**: Gemini 3 Flash: 3,500/3,500 images; Qwen 3.5 122B: 3,193/3,500 images (91.2% coverage; missing images have near-identical GT distribution, so calibration curves are stable).

**Per-model optimal settings** (from prompt arm experiment):

- Gemini 3 Flash: 1024px resolution, scale 1-5 (baseline prompt)
- Qwen 3.5 122B: native resolution (max_pixels=0), scale 1-10 (rescaled to 1-5)

## Test Set Results (n=1,000)

| Model | Method | Overall MAE | Sharp. MAE | Color MAE | wMAE | wSRCC |
|-------|--------|-------------|------------|-----------|------|-------|
| Gemini 3 Flash | raw | 0.7977 | 0.8440 | 0.8326 | 0.8180 | 0.7078 |
| Gemini 3 Flash | linear | 0.2786 | 0.2871 | 0.2808 | 0.2813 | 0.7078 |
| Gemini 3 Flash | 4PL | 0.2755 | 0.2798 | 0.2810 | 0.2780 | 0.7078 |
| Gemini 3 Flash | isotonic | **0.2632** | **0.2610** | **0.2736** | **0.2652** | **0.7159** |
| Qwen 3.5 122B | raw | 1.2317 | 1.2579 | 1.2583 | 1.2449 | 0.7197 |
| Qwen 3.5 122B | linear | 0.2871 | 0.2781 | 0.2862 | 0.2846 | 0.7197 |
| Qwen 3.5 122B | 4PL | 0.2870 | 0.2752 | 0.2860 | 0.2838 | 0.7197 |
| Qwen 3.5 122B | isotonic | **0.2800** | **0.2670** | **0.2767** | **0.2759** | **0.7208** |

**MAE reduction (isotonic vs raw)**: Gemini 67.6%, Qwen 77.8%.

## OOD Generalization (n=520)

| Model | Method | Overall MAE | Sharp. MAE | Color MAE | wMAE |
|-------|--------|-------------|------------|-----------|------|
| Gemini 3 Flash | raw | **0.5400** | **0.5859** | 0.5959 | **0.5655** |
| Gemini 3 Flash | isotonic | 0.6343 | 0.7287 | **0.5235** | 0.6302 |
| Qwen 3.5 122B | raw | **0.8331** | **0.8988** | 0.8051 | 0.8425 |
| Qwen 3.5 122B | isotonic | 0.8391 | 0.9676 | **0.6335** | **0.8198** |

**OOD finding**: Calibration generally *hurts* on OOD data (overall/sharpness worsened), except for color fidelity where it helps. This confirms that ID-fitted calibration curves should only be applied to in-distribution data.

## Model Comparison

| Metric                | Gemini 3 Flash | Qwen 3.5 122B | Winner |
|-----------------------|----------------|---------------|--------|
| Test wSRCC (isotonic) | 0.7159         | 0.7208        | Qwen   |
| Test wMAE (isotonic)  | 0.2652         | 0.2759        | Gemini |
| OOD wMAE (raw)        | 0.5655         | 0.8425        | Gemini |
| Raw bias (overall)    | +0.80          | +1.25         | Gemini |

Gemini has lower absolute error; Qwen has slightly better ranking correlation. Both achieve wMAE < 0.28 after isotonic calibration, which is well within the pseudo-labeling quality threshold.

## Recommendation

The recommended calibration method for the pseudo-labeling pipeline is **isotonic regression** fitted per-model per-dimension on the DIQA training split.

Key findings:

1. **Isotonic regression provides the best MAE reduction** across all model-dimension combinations (67-78%)
2. **SRCC is invariant under monotonic transforms** (as expected) — calibration preserves ranking quality
3. **Calibration should be gated by OOD detection** — applying ID-fitted curves to OOD data degrades performance
4. **Both models reach comparable calibrated accuracy** (wMAE ~0.27) despite very different raw scales, validating per-model calibration as a robust normalization strategy
5. **Qwen's scale-10 prompting** produces higher raw MAE (1.24 vs 0.82) but calibration fully compensates, yielding competitive post-calibration accuracy
