# Research Agenda: Paper 10 --- SigLIP2 Backbone Selection for Teacher-Student DIQA

## Potential Improvements

- **Add DINOv2 and EVA-02 backbones to the probe**: The current experiment only evaluates within the SigLIP2 family. DINOv2 (self-supervised, strong on dense prediction) and EVA-02 (CLIP-pretrained with masked image modeling) may encode quality-relevant structure differently. Expected impact: broader architectural comparison, may reveal that a non-SigLIP2 teacher is optimal. Effort: medium (additional extraction runs, ~15 min GPU per backbone).

- **Multi-layer probe instead of pooler-only**: The current protocol uses only the pooler/mean-pooled CLS output. Concatenating features from multiple transformer layers (e.g., layers 6, 12, 18, 24) captures both low-level texture features (relevant for sharpness) and high-level semantic features (relevant for overall quality). Expected impact: may reveal capacity differences invisible in pooled features alone. Effort: low (modification to extraction, larger Ridge input).

- **Nonlinear probe (2-layer MLP) as upper bound**: Ridge regression measures linear separability, but a 2-layer MLP with ReLU would measure nonlinear separability -- a tighter estimate of fine-tuning potential. If large backbones show minimal linear advantage but large nonlinear advantage, that changes the teacher-student recommendation. Expected impact: resolves whether capacity differences are hidden behind nonlinear feature interactions. Effort: low (sklearn MLPRegressor, no GPU needed).

- **Per-region embedding analysis**: Extract embeddings from spatial subsets of the patch sequence (e.g., center crop tokens vs. margin tokens) to determine whether larger backbones encode more spatially distributed quality information. Expected impact: informs whether attention pooling or region-specific heads should be explored. Effort: medium (requires per-token extraction rather than pooled).

- **Aspect-ratio-preserving padding for fixed-res models**: Before concluding that NaFlex is superior, test whether zero-padding (or reflect-padding) portrait documents to square before feeding to fixed-res models closes the gap. If padding eliminates the NaFlex advantage, the teacher backbone selection opens up to all 9 non-NaFlex variants. Expected impact: potentially significant for teacher selection. Effort: low (preprocessing change, re-extract embeddings).

## Test Refinements

- **Bootstrap confidence intervals on probe SRCC**: The current protocol reports point estimates. Adding 1000-iteration bootstrap resampling on the test set provides 95% CIs per model-dimension pair. Why it matters: small SRCC differences (<0.01) between backbones may not be statistically significant, which changes the teacher recommendation. Estimated effort: minimal (CPU computation on existing predictions).

- **Alpha sensitivity analysis**: Report SRCC across all alpha values rather than only the best. If all models are alpha-insensitive, the representations are robust; if some models are highly alpha-sensitive, their probe SRCC is inflated by overfitting. Why it matters: validates that relative rankings are robust. Estimated effort: negligible (already computed during grid search).

- **Per-image error analysis**: Identify which test images show the largest SRCC contribution differences between base-naflex and the best larger backbone. Visual inspection may reveal whether the gap is driven by specific document types (e.g., dense tables, low-contrast scans). Why it matters: informs whether the gap is systematic or driven by a few outliers. Estimated effort: low (rank-difference analysis on existing predictions).

- **Embedding dimensionality reduction visualization**: t-SNE or UMAP of embeddings colored by MOS quintile for each backbone. If larger backbones produce more quality-separable clusters, this provides visual confirmation of the probe results. Why it matters: communicates findings more intuitively than correlation tables alone. Estimated effort: low (standard visualization).

## Future Experiments

| Experiment | Hypothesis | Data Required | Priority |
|------------|-----------|---------------|----------|
| Fine-tune top teacher candidate on DIQA-5000 | Fine-tuned teacher achieves wSRCC > 0.91 (vs 0.886 for Base) | DIQA-5000, A100 GPU | Critical |
| Dimension-selective teacher evaluation | Different backbones are optimal for different dimensions (e.g., Giant for overall, NaFlex for sharpness) | Probe results from this paper | Critical |
| Aspect-ratio padding vs NaFlex ablation | Zero-padding closes >50% of the NaFlex advantage for fixed-res models | DIQA-5000, re-extraction | High |
| Layout-conditioned fixed-res teacher evaluation | If a non-NaFlex model wins the probe, evaluate whether document layout conditioning improves that teacher's fine-tuned performance | DIQA-5000 + layout masks | High (conditional) |
| Teacher-student distillation end-to-end | Student (Base-NaFlex) trained on teacher pseudo-labels outperforms student trained on human MOS alone | DIQA-5000 + expanded unlabeled documents | High |
| Cross-dataset probe generalization | Backbone rankings on DIQA-5000 hold on an independent document quality dataset | Second document IQA dataset | Medium |
| Probe with frozen vs fine-tuned backbone comparison | Linear probe SRCC rankings predict fine-tuning SRCC rankings (validates probe methodology) | DIQA-5000, multiple fine-tuning runs | Medium |
| NaFlex max_patches sweep (784, 1024, 1568) | Increasing NaFlex token budget improves probe SRCC, with diminishing returns beyond 1024 | DIQA-5000, re-extraction | Medium |

## Conditional Research: Layout Awareness for Fixed-Resolution Teachers

**Trigger**: Paper 10 probe results show a non-NaFlex model outperforming both NaFlex variants by >2% wSRCC.

**Context**: Only 2 of 11 SigLIP2 variants support NaFlex (base-p16-naflex, so400m-p16-naflex). If probe results favor a fixed-resolution model (e.g., giant-opt-p16-384 at 1B params), that teacher will compress portrait documents (~1.4:1) into a 384px square, losing aspect ratio and fine-grained spatial information. The question becomes whether document structure awareness can partially recover this loss during teacher fine-tuning.

**Important**: DocIQ's Layout Fusion Downsampler is CNN-specific (conv parallel paths fusing layout masks with RGB before ResNet-50). It does not apply to ViT architectures. The relevant interventions for ViT-based teachers are architecturally distinct.

**Evaluation sequence (ordered by simplicity)**:

1. **Aspect-ratio-preserving padding** (simplest): Zero-pad or reflect-pad portrait images to square before resize. No layout model needed. If this closes the gap vs NaFlex, no further work is needed.

2. **Layout-guided cropping/tiling**: Use DocLayout-YOLO masks to select semantically important 384x384 crops (e.g., centered on text regions), process multiple crops, and aggregate embeddings. Preserves full resolution in critical regions.

3. **Layout token concatenation**: Append learned layout class embeddings (from DocLayout-YOLO detections) to the ViT patch token sequence. Lightweight, ~11 additional tokens. Requires fine-tuning to learn the layout-quality relationship.

4. **Layout-conditioned attention bias**: Use layout masks to generate per-head attention biases (e.g., encourage within-region attention). Most complex, but directly injects document structure into the transformer computation.

**Reference data**: DocIQ ablation showed layout fusion contributed only -0.0115 avg SRCC on ResNet-50 (smallest component, vs -0.0323 for multi-scale feature fusion). The gain on ViT may be smaller still. This sets expectations: layout conditioning is unlikely to yield >1% improvement and should only be pursued if simpler interventions (padding, tiling) are insufficient.

**Decision point**: If padding alone recovers >80% of the NaFlex advantage, skip layout conditioning entirely. The implementation complexity is not justified for marginal gains on the teacher (whose purpose is to generate pseudo-labels, not serve as a production model).

## Peer Review Feedback Log

| Date | Reviewer | Category | Feedback | Status |
|------|----------|----------|----------|--------|
| | | | | |
