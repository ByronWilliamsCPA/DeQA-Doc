# Next Iteration: OOD Detector Training Data Expansion Strategy

> **Date**: 2026-03-06
> **Status**: Analysis Complete — 13-Model Consensus Evaluation
> **Models Consulted**: GPT-5.2, Gemini 3.1 Pro, Gemini 3 Flash, DeepSeek V3.2, Minimax M2.5,
> Grok 4.1 Fast, Qwen3.5-397B, Qwen3.5-Plus, Kimi K2.5, GLM-5, Arcee Trinity, Nemotron Nano 9B, GLM-4.5 Air

## Executive Summary

The current Tier 1 OOD detector achieves AUROC 0.9963 on synthetic data, but this metric is
misleading. All 13 models consulted unanimously agree on three critical findings:

1. **Real-world OOD evaluation must happen before any expansion** — synthetic-only testing creates
   dangerous overconfidence
2. **The circular training problem is real and severe** — SigLIP2 pseudo-labels cannot improve
   SigLIP2's own blind spots
3. **The checkpoint mismatch (445 missing keys) must be fixed first** — the 8-unit train/test
   distance shift invalidates all threshold calibration

The highest-impact, most feasible path forward is: **extract embeddings from public document
datasets (RVL-CDIP, Tobacco800) as evaluation sets first**, then selectively expand the covariance
model with domain-validated documents.

---

## 1. Current State Assessment

### What Works

| Metric | Value | Notes |
|--------|-------|-------|
| AUROC (synthetic OOD) | 0.9963 | Near-perfect on 13 synthetic categories |
| OOD TPR at 5% FPR | 99.5% | Threshold = 46.0 (test p95) |
| Inference latency | ~1-2ms | Negligible addition to SigLIP2 forward pass |
| Training data | 4,400 embeddings | DIQA-5000 train+val, 768-dim SigLIP2 |

### What Doesn't Work (or Is Untested)

| Issue | Severity | Consensus |
|-------|----------|-----------|
| No real-world OOD evaluation | **Critical** | 13/13 models |
| Checkpoint mismatch (8-unit train/test shift) | **Critical** | 13/13 models |
| Single-source training data (DIQA-5000 only) | **High** | 13/13 models |
| Synthetic OOD categories are "easy" (far from boundary) | **High** | 12/13 models |
| No document type/script diversity metadata | **Medium** | 10/13 models |
| Quality distribution skew (89% good/fair) | **Medium** | 8/13 models |
| No "Not-a-Document" class (desks, fingers, screenshots) | **Medium** | 3/13 models |

### The Circular Training Problem

The pseudo-labeling pipeline uses SigLIP2 to generate quality scores for unlabeled documents. If
these pseudo-labels are fed back to expand SigLIP2's training set, the system enters a
self-reinforcing bias loop:

```
SigLIP2 predicts quality → labels added to training → SigLIP2 learns its own biases
    ↑                                                           ↓
    └───────────── same blind spots reinforced ←────────────────┘
```

**All 13 models confirm this is a real problem.** The detector cannot bootstrap out of its own blind
spots without an external signal. However, the severity depends on the use case:

- **For OOD detection (embedding covariance fitting)**: Labels are NOT needed — only diverse
  embeddings matter. The circular problem is less severe here.
- **For IQA quality prediction on new document types**: External labels (human or calibrated VLM)
  are essential. The circular problem is fully severe here.

---

## 2. Critical Prerequisite: Fix Before Anything Else

### 2a. Resolve Checkpoint Mismatch

The current SigLIP2 checkpoint has 445 missing keys and 368 unexpected keys, causing an ~8-unit
Mahalanobis distance shift between train (median 24.1) and test (median 32.6) distributions.

**Impact**: Every threshold, every AUROC comparison, and every GMM cluster will be noisy until this
is resolved. The production threshold of 46.0 is calibrated against this shifted distribution — it
may be too aggressive or too conservative with a properly matched checkpoint.

**Action**: Identify and load the correct checkpoint weights before any data expansion work.

### 2b. Establish Real-World OOD Baseline

Before implementing ANY improvement, evaluate the current detector on real document datasets:

```bash
# Step 1: Download and prepare datasets
# - RVL-CDIP: 400K documents, 16 categories (sample 500-1000)
# - Tobacco800: 1,600 scanned documents
# - CORD: 11K receipt images (sample 500)

# Step 2: Extract SigLIP2 embeddings
python3 scripts/extract_siglip2_embeddings.py \
    --checkpoint <fixed_checkpoint> \
    --meta-path rvl_cdip_sample.json \
    --image-root /path/to/rvl_cdip/ \
    --output /path/to/rvl_cdip_embeddings.npy

# Step 3: Compute Mahalanobis distances against current detector
# Step 4: Analyze: What percentage fall above/below threshold 46.0?
# Step 5: Manually inspect boundary cases (d_M = 40-55)
```

**Expected outcome**: AUROC on real datasets will likely drop to 0.70-0.85 range (vs 0.9963 on
synthetic). This will reveal which document types/categories are hardest to distinguish.

---

## 3. Proposed Options — Consensus Evaluation

### Option Rankings (13-Model Aggregate)

| Rank | Option | Impact | Feasibility | Effort | Consensus Support |
|------|--------|--------|-------------|--------|-------------------|
| **1** | **Public datasets as evaluation + covariance expansion** | Very High | Very High | 1 week | 13/13 |
| **2** | **Dual-embedding OOD (SigLIP2 + doc model)** | High | Medium | 2 weeks | 9/13 |
| **3** | **VLM committee labeling (with calibration)** | High | Medium | 2-3 weeks | 11/13 (with caveats) |
| **4** | **Active learning (BALD) on boundary cases** | High | Medium | 4+ weeks | 10/13 |
| **5** | **Controlled degradation of real documents** | Medium-High | High | 1-2 weeks | 7/13 |
| **6** | **GMM clustering** | Medium | High | 0.5 weeks | 5/13 (after expansion) |
| **7** | **Human annotation** | Very High quality | Low | 8+ weeks | 6/13 (targeted only) |
| **8** | **Pure synthetic generation** | Low-Medium | Medium | 3 weeks | 2/13 (supplement only) |

### Detailed Option Analysis

#### Option 1: Public Document Datasets (HIGHEST PRIORITY)

**Unanimous consensus (13/13)**: This is the single most impactful and feasible improvement.

**Key insight (GPT-5.2, Gemini 3.1 Pro)**: There are TWO different ways to use public datasets, and
they serve different goals:

| Use Case | What You Need | Goal |
|----------|---------------|------|
| **Evaluation** (do this FIRST) | Embeddings only | Measure real-world AUROC, find blind spots |
| **Covariance expansion** (do this SECOND) | Embeddings only | Broaden "in-distribution" definition |

**Critical warning (GPT-5.2)**: Expanding the covariance model with RVL-CDIP embeddings makes those
document types "in-distribution" by definition. This **collapses the OOD gate** — fewer documents
trigger Tier 2. Only do this if you've validated that SigLIP2's IQA predictions are reliable on
those document types.

**Two conflicting OOD goals** (GPT-5.2, GLM-5):

1. **"Not DIQA-5000-like"** → Do NOT expand covariance. Keep the gate tight.
2. **"SigLIP2 prediction unreliable"** → Expand covariance ONLY where IQA is validated.

**Recommended datasets**:

| Dataset | Size | Categories | Priority |
|---------|------|------------|----------|
| RVL-CDIP | 400K (sample 5-10K) | 16 doc types | High — most diverse |
| Tobacco800 | 1,600 | Scanned industry docs | High — real scan artifacts |
| CORD | 11K | Receipts | Medium — structured layouts |
| DocVQA | 12K+ | Mixed documents | Medium — diverse content |
| FUNSD | 199 | Noisy scanned forms | Low — small but targeted |

**Implementation**:

```python
# 1. Extract embeddings (no labels needed)
embeddings_rvl = extract_embeddings(rvl_cdip_images)  # shape: (N, 768)

# 2. Evaluate current detector
distances = detector.score_batch(embeddings_rvl)
auroc = compute_auroc(diqa_test_distances, rvl_distances)
# → This tells you how well the current detector separates DIQA from RVL-CDIP

# 3. If SigLIP2 IQA is validated on these docs, selectively expand:
expanded_embeddings = np.vstack([diqa_train_embeddings, validated_rvl_embeddings])
new_detector = EmbeddingOODDetector.from_embeddings(expanded_embeddings)
```

#### Option 2: Dual-Embedding OOD Detection (HIGH IMPACT)

**Strong support (9/13)**: Use a document-specialized model alongside SigLIP2.

**Rationale (Gemini Flash, Minimax M2.5)**: SigLIP2's embedding space is optimized for image quality
features, not document structure. A document-specialized model (DiT, LayoutLMv3, Donut) encodes
layout, script type, and structural features that SigLIP2 may collapse.

**Architecture**:

```
Input Image
    ├──→ SigLIP2 (768-dim)  ──→ Mahalanobis d₁ ──→ IQA-aware OOD
    │
    └──→ DiT/LayoutLMv3 (768-dim) ──→ Mahalanobis d₂ ──→ Structure-aware OOD
                                              │
                                    d_hybrid = α*d₁ + (1-α)*d₂
                                    OR: flag if d₁ > t₁ OR d₂ > t₂
```

**Practical variant (GPT-5.2)**: Don't concatenate embeddings (curse of dimensionality). Instead,
compute separate Mahalanobis distances in each space, then gate on `(d₁ > threshold₁ OR d₂ >
threshold₂)`. This catches failures in either space independently.

**Candidate document models**:

| Model | Embedding Dim | Pre-training Data | Inference Cost |
|-------|---------------|-------------------|----------------|
| DiT-Base | 768 | IIT-CDIP (42M pages) | ~30ms |
| LayoutLMv3-Base | 768 | IIT-CDIP (11M pages) | ~25ms |
| Donut | 1024 | SynthDoG (1.2M docs) | ~50ms |
| DINOv2-Base | 768 | LVD-142M (diverse) | ~15ms |

**Effort**: ~2 weeks for implementation + evaluation. One-time cost, then negligible runtime
addition.

#### Option 3: VLM Committee Labeling (WITH MAJOR CAVEATS)

**Supported with strong caveats (11/13)**.

**Critical warning (Gemini 3.1 Pro, 9/10 confidence)**: VLMs assess *readability and semantics*,
NOT low-level optical quality. A blurry document that's still readable will get "good" from a VLM
but "poor" from human IQA raters. Using VLM labels as ground truth for IQA will corrupt the dataset
with labels that reflect readability rather than true optical quality.

**When VLM committee IS appropriate**:
- Labeling "easy" cases where quality is unambiguous (pristine or severely degraded)
- Breaking ties between document types (is this a form or a letter?)
- Providing coarse quality categories as a pre-filter

**When VLM committee is NOT appropriate**:
- Fine-grained quality assessment (good vs. fair)
- Documents with subtle degradation (slight blur, minor compression artifacts)
- Non-Latin scripts (VLMs may have systematic biases)

**If proceeding, the protocol must include**:

1. **Calibration first**: Run 3+ VLMs on 200-500 DIQA-5000 samples with known human labels.
   Measure correlation (expect 0.6-0.8, not 0.9+).
2. **Minimum 3 diverse models**: Qwen3-VL-8B + GPT-4o + Gemini Flash (architecturally diverse).
3. **Abstention on disagreement**: If cross-model JSD > 0.1 or std > 0.5 on 5-point scale, discard
   the sample. Do NOT average disagreeing labels.
4. **Human bridge set**: 50-100 human-labeled images spanning diverse scripts/layouts to validate
   VLM labels before scaling.
5. **Progressive trust**: Start with high-agreement easy cases, gradually expand.

#### Option 4: GMM Clustering (DEPRIORITIZED)

**Consensus (10/13): GMM should NOT come before diversity expansion.**

On the current homogeneous DIQA-5000 data (89% good/fair, single source), clustering will likely
separate quality levels (good vs fair vs poor), not document types. This is the wrong axis for OOD
detection.

**When to use GMM**: After Options 1-3 have expanded the embedding distribution with diverse
document types. Then GMM can model per-type distributions (letters, receipts, forms, etc.) and
detect OOD within each type.

**If implementing early**: Use a representation less entangled with IQA — earlier SigLIP2 layers or
a document model's embeddings — so clusters reflect structure, not quality.

#### Option 5: Controlled Degradation of Real Documents (NEW — FROM CONSENSUS)

**Novel approach proposed by Gemini 3.1 Pro (7/13 support)**:

Instead of generating synthetic *documents*, apply synthetic *degradations* to real documents from
public datasets:

```python
# Take pristine documents from RVL-CDIP
pristine_doc = load_image("rvl_cdip/letter_001.jpg")

# Apply controlled, mathematically defined degradations
degraded_versions = [
    apply_gaussian_blur(pristine_doc, sigma=2.0),   # known quality: poor (blur)
    apply_jpeg_compression(pristine_doc, quality=10), # known quality: bad (compression)
    apply_gaussian_noise(pristine_doc, std=25),       # known quality: poor (noise)
    apply_resolution_reduction(pristine_doc, scale=0.25), # known quality: fair (low-res)
]
# Ground truth quality is KNOWN from the degradation parameters
# No VLM or human annotation needed
```

**Advantages**:
- Real document *semantics* + controlled quality *degradation* = cheap, reliable ground truth
- Eliminates the synthetic-to-real domain gap for document content
- Degradation parameters directly map to quality levels
- Unlimited volume at near-zero cost

**Limitations**:
- Only captures degradation-type quality variation, not inherent document quality
- Requires calibrating degradation parameters against DIQA-5000 MOS scale

#### Option 6: Active Learning (AFTER DEPLOYMENT)

**Strong support (10/13) but requires production deployment first.**

The existing BALD-based active learning module is already implemented. The optimal strategy:

1. Deploy detector in **shadow mode** alongside production
2. Log all documents with d_M in the boundary zone (40-55)
3. Prioritize for human annotation: high BALD + near boundary
4. Use **adaptive rater counts** (Dawid-Skene/MACE aggregation) — not 15 raters per image
5. Re-fit detector iteratively as annotations accumulate

#### Option 7: Pure Synthetic Generation (SUPPLEMENT ONLY)

**Weak support (2/13 as primary strategy)**: Synthetic documents are useful for extreme gaps (rare
scripts, unusual layouts) but should NOT be the primary path to real-world OOD robustness.

**Known issues**:
- Sim-to-real domain gap is significant for documents
- Synthetic docs lack real capture artifacts (moire, toner bleed, folding marks, scanner dust)
- Current synthetic OOD (370 images) already proved the detector works on far-OOD — more
  synthetic data won't help with near-boundary detection

---

## 4. Novel Approaches Identified by Consensus

Several approaches were proposed that weren't in the original analysis:

### 4a. PCA Dimensionality Reduction (Gemini Flash)

768 dimensions with 4,400 samples is a tight ratio even with Ledoit-Wolf shrinkage. PCA to 128-256
dimensions before fitting Mahalanobis could:
- Remove noise dimensions that capture identity rather than distribution
- Improve OOD robustness by focusing on principal directions of variation
- Reduce computation (though already negligible at 1-2ms)

### 4b. ODIN/Energy-Based Ensemble (Grok, DeepSeek)

Combine Mahalanobis distance with energy-based OOD scores computed from SigLIP2's logits or
softmax outputs. Multiple complementary OOD signals catch different failure modes:

```
OOD_score = w₁ * mahalanobis(embedding) +
            w₂ * energy_score(logits) +
            w₃ * knn_density(embedding, k=10)
```

Estimated improvement: +5-10% AUROC on near-boundary OOD with no additional data.

### 4c. "Not-a-Document" Class (Gemini Flash)

Real-world deployments receive non-document images (desk photos, fingers over camera, screenshots,
blank pages). None of the current options address this. A simple binary classifier or additional OOD
category for non-documents would prevent quality scores on invalid inputs.

### 4d. Conformal Prediction / Risk-Controlled Gating (GPT-5.2, Kimi K2.5)

Instead of optimizing AUROC, use conformal prediction to guarantee a maximum error rate under
exchangeability assumptions. This is more operationally meaningful — you can promise "we'll catch
95% of unreliable predictions" rather than "our AUROC is 0.99."

### 4e. Spectral Gap Analysis (Kimi K2.5)

PCA on current embeddings to identify which principal components have variance gaps, then
specifically target those dimensions with new data. This systematically identifies WHERE the
embedding space is sparse rather than randomly adding diverse data.

### 4f. Test-Time Augmentation (Gemini 3.1 Pro)

Run SigLIP2 on multiple augmented versions (crops, rotations, slight color shifts) of the same
input. High variance in predictions indicates epistemic uncertainty (potential OOD) without any
additional models or data.

### 4g. Document-Type Classifier as Auxiliary Signal (Minimax M2.5)

Train a 16-category document classifier (using RVL-CDIP categories) on top of SigLIP2 embeddings.
Classifier entropy serves as an OOD signal — if the classifier is uncertain about document type,
the image may be outside the training distribution.

---

## 5. Recommended Implementation Plan

### Phase 0: Prerequisites (Week 1)

- [ ] **Fix checkpoint mismatch** — resolve 445 missing / 368 unexpected keys
- [ ] **Re-fit OOD detector** with corrected checkpoint
- [ ] **Re-calibrate thresholds** — train/test gap should close from ~8 units to ~2-3

### Phase 1: Real-World Evaluation (Week 1-2)

- [ ] Download RVL-CDIP (sample 5,000 across 16 categories), Tobacco800, CORD (sample 500)
- [ ] Extract SigLIP2 embeddings from all datasets
- [ ] Compute Mahalanobis distances against current detector
- [ ] Analyze per-category detection rates (which doc types look in-distribution?)
- [ ] Manually inspect boundary cases (d_M = 40-55)
- [ ] **Decision point**: Does real-world performance warrant further investment?

### Phase 2: Quick Wins (Week 2-3)

- [ ] Implement PCA dimensionality reduction (768 → 256) and compare AUROC
- [ ] Implement kNN density as complementary OOD signal
- [ ] Test energy-based / ODIN ensemble approach
- [ ] If IQA validated on specific doc types: selectively expand covariance
- [ ] Begin controlled degradation pipeline (blur/noise/compression on RVL-CDIP pristine docs)

### Phase 3: Dual-Embedding OOD (Week 3-5)

- [ ] Extract DiT or LayoutLMv3 embeddings from DIQA-5000 + public datasets
- [ ] Fit separate Mahalanobis detector in document-model embedding space
- [ ] Implement dual-gate: flag OOD if either space detects anomaly
- [ ] Evaluate improvement on real-world datasets from Phase 1

### Phase 4: VLM Consensus Labeling (Week 4-6, if needed)

- [ ] Calibrate 3 VLMs against DIQA-5000 human labels (200-500 samples)
- [ ] Measure VLM-human correlation per quality dimension
- [ ] If correlation > 0.7: label high-agreement "easy" cases from public datasets
- [ ] Validate VLM labels on 50-100 human-annotated bridge set
- [ ] Add validated consensus labels to training pool

### Phase 5: Active Refinement (Ongoing, post-deployment)

- [ ] Deploy detector in shadow mode
- [ ] Log boundary cases (d_M = 40-55) and high-BALD samples
- [ ] Targeted human annotation on maximally informative samples
- [ ] Iterative re-fitting as annotations accumulate

---

## 6. Key Decision: What Does "OOD" Mean for This System?

GPT-5.2 identified the most fundamental question that must be answered before proceeding:

> **Goal A: "Not DIQA-5000-like"** (pure distribution shift detection)
> - Keep the gate tight. Do NOT expand covariance.
> - Any document not from DIQA-5000's distribution triggers Tier 2.
> - Simple, conservative, high Tier 2 API costs.
>
> **Goal B: "SigLIP2 prediction unreliable"** (operational risk gating)
> - Expand covariance WHERE SigLIP2 IQA is validated.
> - Only trigger Tier 2 for truly unreliable predictions.
> - More complex, requires IQA validation per domain, lower API costs.

**These goals imply opposite strategies for data expansion.** Options 1, 3, 5, and 8 all assume
Goal B. If Goal A is the correct objective, most expansion options are counterproductive.

**Recommendation**: Goal B is more operationally useful but requires a validation framework.
Before adding any new document type to "in-distribution," validate SigLIP2's IQA predictions on
that type against human labels or calibrated VLM consensus.

---

## 7. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real-world AUROC much lower than synthetic | High | Critical | Phase 1 evaluation reveals this early |
| Covariance expansion collapses OOD gate | Medium | High | Use evaluation-first approach; don't blindly expand |
| VLM labels corrupt IQA training | Medium | High | Calibrate against human labels; abstain on disagreement |
| Checkpoint fix doesn't close train/test gap | Low | Medium | Re-evaluate whether Ledoit-Wolf shrinkage is sufficient |
| Dual-embedding adds unacceptable latency | Low | Medium | DiT-Base adds ~30ms; benchmark first |
| Near-boundary OOD remains undetectable | Medium | High | Dual-embedding + ensemble approach is best mitigation |

---

## 8. Model-by-Model Consensus Summary

| Model | Confidence | Top Priority | Key Unique Insight |
|-------|-----------|--------------|-------------------|
| GPT-5.2 | 8/10 | Fix calibration + real eval | Two conflicting OOD goals require explicit choice |
| Gemini 3.1 Pro | 9/10 | Fix checkpoint + real eval | VLMs assess readability, NOT optical quality |
| Gemini 3 Flash | 9/10 | Option 7 + Option 1 | PCA reduction; "Not-a-Document" class blind spot |
| DeepSeek V3.2 | 8/10 | Real eval + Option 1+3 | Temperature-scaled confidence; ensemble detectors |
| Minimax M2.5 | 8/10 | Option 1 (no labels needed) | Doc-type classifier as auxiliary OOD signal |
| Grok 4.1 Fast | 9/10 | Option 1 → 3 → 4 | ODIN/energy ensemble +5-10% AUROC; FAISS clustering |
| Qwen3.5-397B | 8/10 | Option 1+3+6 core | GMM mispositioned; kNN density as alternative |
| Qwen3.5-Plus | 8/10 | Option 1 highest priority | GMM potentially misleading on homogeneous data |
| Kimi K2.5 | 8/10 | Unlabeled diversity first | Spectral gap analysis; conformal prediction |
| GLM-5 | 8/10 | Real eval before expansion | "Boundary patrol" (d_M 40-55); two problems conflated |
| Arcee Trinity | 7/10 | Option 3 risk-adjusted | Progressive trust building for VLM labels |
| Nemotron Nano | 7/10 | Option 1 + Option 3 | Cascade validation (VLM → human sample → scale) |
| GLM-4.5 Air | 7/10 | Hybrid (Option 8) | Progressive trust; active learning post-deployment |

### Unanimous Agreements (13/13 models)

1. Fix the checkpoint mismatch before any data expansion work
2. Evaluate on real-world OOD datasets before implementing improvements
3. The circular training problem is real — external signal is required
4. Option 1 (public datasets) is the highest-priority action

### Strong Consensus (10+ models)

5. GMM clustering should NOT precede diversity expansion (10/13)
6. VLMs are unreliable for fine-grained IQA — need calibration and abstention (11/13)
7. Minimum 3 diverse VLM models for committee labeling (12/13)
8. Active learning is high-value but requires deployment first (10/13)

### Areas of Disagreement

- **Option 7 (dual-embedding)**: 9/13 support, but 4 models consider it too complex for the
  benefit. The OR-gate variant (separate thresholds per space) resolves most complexity concerns.
- **VLM committee viability**: Gemini 3.1 Pro strongly opposes using VLMs for IQA labeling (VLMs
  measure readability, not optical quality). Others support it with calibration. The truth likely
  depends on the quality dimension — VLMs may be adequate for "overall quality" but poor for
  "sharpness" and "color fidelity."
- **GMM utility**: 5/13 support GMM even on current data; 8/13 say it's pointless on homogeneous
  data. The compromise: implement GMM after Phase 1 expansion.

---

## Appendix: Consensus Methodology

This analysis was produced through a structured multi-model consensus process:

1. **Initial analysis** prepared by Claude Opus 4.6 based on thorough codebase exploration
2. **13 models consulted** via OpenRouter API with identical prompts and full project context
3. **Each model provided**: verdict, per-option analysis, ranking, confidence score, key takeaways
4. **Synthesis** performed by aggregating agreements, disagreements, and novel insights across all
   13 responses

The models received the full OOD detector README, CLAUDE.md project context, and the complete
8-option proposal with 6 evaluation questions. Average confidence across all models: **8.2/10**.
