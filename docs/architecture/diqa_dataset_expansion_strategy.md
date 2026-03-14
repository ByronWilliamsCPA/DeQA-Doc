# DIQA Dataset Expansion Strategy: Staged Pseudo-Label Pipeline

**Author:** Byron Williams
**Date:** March 2026
**Status:** Draft — Pending Review
**Dependencies:** Paper 2 (Cross-Domain), Paper 7 (Pseudo-Labeling), SigLIP2-IQA v2

---

## Executive Summary

The DIQA-5000 dataset has reached its training ceiling. SigLIP2-IQA achieves 0.886 MainScore
on DIQA-5000 but drops 30% to 0.620 on OOD documents. Analysis reveals **six critical gaps**
in the current training data that drive this degradation: quality distribution imbalance,
script homogeneity, layout narrowness, capture method bias, degradation mode coverage, and
processing-state blindness (binarized/DPI-extreme documents).

This document defines a **4-tier staged expansion** from 350 real images (3,500 with 10x
synthetic augmentation) to ~20,000+ effective training images, using VLM pseudo-labels where
reliable, controlled synthetic degradation from the synth-multiscript-v3 pipeline where
deterministic labels are possible, and alternative strategies where VLMs fail. Each tier
has explicit validation gates, rollback plans, and holdout sets. Total VLM cost: ~$40.
Total compute: ~12-15 GPU-days on 4x A100.

**Key resource**: The synth-multiscript-v3 pipeline (190K pristine base images, 27 scripts,
8 IQA dimensions, deferred degradation with Augraphy + Albumentations) provides controlled
degradation with **deterministic quality labels derived from degradation parameters** — no
VLM annotation needed for synthetic samples. This is the primary source for quality-tail
correction and degradation hardening.

**Validation strategy**: Datasets with text ground truth are prioritized to enable
**OCR-based quality validation** as an independent cross-check — OCR error rates should
correlate with quality scores, providing a signal orthogonal to both VLM judgments and
SigLIP2 predictions.

**Critical constraint**: Three document categories (binarized, pristine, DPI extremes) are
**excluded from pseudo-labeling entirely** due to negative or near-zero VLM SRCC. These
require human annotation or domain-specific quality metrics in a future phase.

---

## 1. Current State Assessment

### 1.1 Training Data Composition

| Attribute | Current Distribution | Problem |
| --------- | -------------------- | ------- |
| **Real base images** | **350 train** / 150 val+test (sacred, never violated) | Tiny real-image base |
| **Effective train** | 3,500 (350 base x 10 synthetic augmentations) | Single-source augmentation |
| **Script** | 75% Hans, 23% Latin, 2% other | No Arabic, Devanagari, Cyrillic, etc. |
| **Quality** | 61% fair, 14.6% good, 18% poor, 5.6% bad, **0.5% excellent** | Extreme tail starvation |
| **Domain** | EDU 41%, SCI 31%, TEC 25% | No forms, invoices, receipts, legal |
| **Capture** | 91% synthetic, 9% camera | No scanner, no mobile diversity |
| **Content** | 64% formula, 53% figure, 21% handwriting | Academic bias |
| **Layout** | Unknown (no metadata) | No form/tabular/multi-column tracking |

> **Sacred constraint**: The DIQA-5000 val+test split (150 base images → 1,500 with
> synthetic augmentation) must **never** be used for training under any circumstance.
> Only the 350 real base training images and their 10x synthetic augmentations (3,500
> effective samples) are available as the training foundation.

### 1.2 Model Performance by OOD Category

From Paper 2 (21 VLMs + 3 fine-tuned models on 520-image synthetic OOD benchmark):

| Tier | Categories | Best VLM SRCC | SigLIP2 SRCC | Pseudo-Label Viable? |
| ---- | ---------- | ------------- | ------------ | -------------------- |
| **A** | Non-Latin scripts (Tibetan, Myanmar, Ethiopic) | 0.73–0.85 | -0.08–0.49 | Yes, with caveats |
| **A** | Adversarial typefaces (Nastaliq, Fraktur) | 0.76–0.85 | 0.04–0.60 | Yes |
| **A** | CJK vertical, multiscript | 0.62–0.81 | -0.36–0.77 | Yes, marginal for CJK |
| **B** | Form layouts | 0.17–0.33 | -0.09 | No — VLMs unreliable |
| **B** | Heavily degraded | 0.17–0.24 | -0.05 | No — VLMs unreliable |
| **C** | Binarized | -0.34 to -0.49 | -0.50 | **Absolutely not** |
| **C** | Pristine | -0.09 to +0.30 | -0.004 | No — no quality signal |
| **C** | DPI extremes (very low, very high) | -0.41 to +0.25 | 0.07–0.14 | No — anti-correlated |

### 1.3 Key Research Findings Constraining the Strategy

1. **Soft-label training preserves generalization.** DeQA-Doc-3Specialists (soft-label loss)
   shows delta = -0.001 ID-to-OOD vs. HyperIQA++ (regression loss) at delta = -0.205.
   All expansion data must use soft-label distributions, not point estimates.

2. **Two-model VLM consensus outperforms singles.** Gemini 3.1 Flash Lite (combined prompt)
   paired with Qwen 3.5-122B achieves wSRCC = 0.777, exceeding either alone (0.743, 0.728).
   Residual correlation r=0.461 confirms complementary error patterns. This is the labeling
   backbone for all tiers. Cost: ~$0.002/image.

3. **Synthetic OOD SRCCs are upper bounds.** Measured on controlled synthetic data with
   uni-dimensional quality variation. Real-world per-category reliability is unknown and
   likely lower. All expansion tiers must validate against held-out data.

4. **VLM calibration may not transfer to OOD.** Isotonic calibration fitted on DIQA-5000
   corrects for systematic positive bias (+0.5–1.5 MOS). On unfamiliar document types,
   VLMs may have different or reversed bias profiles.

### 1.4 Available Synthetic Generation Infrastructure

The sister project (`/home/byron/dev/image_detection/`) provides mature synthetic data
generation infrastructure that this expansion plan leverages heavily.

#### Synth-Multiscript-v3 Pipeline

**190,485 pristine base images** across 27 scripts, stored at
`gs://image_detection_b/synth_multiscript_v3/`. Key design: **pristine base + deferred
degradation** — base images are stored clean, with degradation parameters recorded in
metadata sidecars for reproducible replay. This is ideal for DIQA expansion because:

- **Deterministic IQA labels**: Quality scores derived directly from degradation parameters
  (blur_severity = sigma/5.0, noise_severity = amount/0.05, overall_quality = 1.0 - max(...)).
  No VLM annotation needed. Perfect ground truth at zero labeling cost.
- **27-script coverage**: Arab, Armn, Beng, Cyrl, Deva, Ethi, Geor, Grek, Gujr, Guru,
  Hans, Hant, Hebr, Jpan, Khmr, Knda, Kore, Laoo, Latn, Mlym, Mymr, Orya, Sinh, Taml,
  Telu, Thai, Tibt — directly addresses the script homogeneity gap.
- **CJK vertical text**: Jpan 30%, Hans 10%, Hant 10% validated vertical text samples.
- **Layout diversity**: 11 generator types mapped to 4 Layer 2 types (single_column 64%,
  multi_column 25%, form_based 8%, complex 3%).
- **Quality tier distribution**: PRISTINE 10%, HIGH 25%, MEDIUM 35%, LOW 20%, DEGRADED 10%.
- **7-tier DPI**: 72, 100, 150, 200, 300, 400, 600 DPI with known character heights.
- **Hybrid augmentation**: Augraphy (document-specific: bleed-through, ink degradation,
  paper aging, bookbinding, dirty drum, folding) + Albumentations (general: blur, noise,
  compression, color jitter).

**Distribution caveat**: Arab is at 3.8x target (49K), 17 scripts below target. Requires
rebalancing before use. See synth-multiscript-v3.md for rebalancing protocol.

#### Curriculum Learning Methodology

The `SYNTHETIC_REAL_TRAINING_METHODOLOGY.md` defines a proven 4-stage curriculum:

1. **Synthetic Foundation** — broad coverage, controlled quality (synth-multiscript-v3)
2. **Mixed Training** — 70% synthetic + 30% real, weighted sampling
3. **Real Document Fine-tuning** — domain-specific refinement on human-labeled data
4. **Active Learning** — production feedback loop with uncertainty-triggered annotation

This curriculum directly maps to our tiered expansion: Tier 1 (synthetic foundation for
quality tails) → Tier 2 (mixed with real script diversity) → Tier 3 (degradation hardening)
→ Tier 4 (active learning refinement).

#### v4 Font Diversity & Adversarial Strategy

The v4 font strategy adds **5 adversarial attack vectors** with 14 downloaded adversarial
fonts across 11 scripts. The tiered font sampling (SYSTEM 40%, REGIONAL 25%, STYLISTIC 15%,
HANDWRITING 15%, ADVERSARIAL 5%) ensures synthetic documents don't overfit to standard
Noto typefaces. Key adversarial fonts relevant to DIQA:

- **UnifrakturMaguntia** (Fraktur/Blackletter) — addresses adversarial typeface gap
- **Jaini/Modak** (broken shirorekha Devanagari) — structural feature destruction
- **LiuJianMaoCao** (grass script Chinese) — calligraphic transfer
- **ReemKufi** (geometric Kufic Arabic) — no cursive flow, resembles Latin
- **Lobster/Pacifico** (cross-script Cyrillic/Latin) — confusion pairs

These fonts create documents where quality assessment is genuinely harder — the model
must learn quality from degradation signals, not font familiarity.

#### Thousand Character Classic Dataset (NEW)

**391 historical CJK calligraphy images** spanning the Sui dynasty (6th century) to Qing
dynasty (19th century), covering 6 major Chinese script styles across 3 CJK writing
traditions (Chinese 73%, Japanese 15%, Korean 12%).

**Value for DIQA expansion:**

- **Historical degradation diversity**: Foxing, ink fading, paper aging, staining, worm
  damage — real degradation patterns absent from DIQA-5000's synthetic augmentation.
- **Legibility-to-quality mapping**: Script styles have known legibility ratings (kaishu
  GOOD 0.75, caoshu FAIR 0.45, kuangcao POOR 0.25) — provides partial quality prior.
- **CJK vertical text**: 100% top-to-bottom, right-to-left column order — directly fills
  the CJK vertical layout gap (current SigLIP2 SRCC = -0.363).
- **Variable scan quality**: Museum high-resolution scans mixed with web-quality photographs
  provides natural capture method diversity.
- **Tier_0_exact text content**: Known fixed literary text enables text_content L2 fields
  without OCR — unique provenance advantage.

**Limitation**: Only 391 images. Best used as a quality-diverse seed for CJK historical
documents rather than a primary training source. VLM pseudo-labels for quality assessment
are viable (calligraphy falls under the "adversarial typefaces" Tier A category, SRCC
0.76-0.85).

---

## 2. Gap Analysis: Ranked by Production Impact

Synthesized from multi-agent critical evaluation. Ranked by real-world deployment risk,
not academic completeness.

### Gap 1: Quality Distribution Imbalance (CRITICAL)

**The problem:** 61% of training data is "fair" quality. The model has almost no signal
for excellent (0.5%) or bad (5.6%) quality levels. Production users care most about
extremes: "is this usable?" and "is this archival quality?"

**Evidence:** Pristine SRCC = -0.09, heavily degraded SRCC = 0.24. The model literally
cannot rank quality at the tails — worse than random for pristine documents.

**Impact:** Users submit high-quality scans and get random scores. Highest source of
user complaints and trust erosion.

**Strategy:** Controlled degradation (deterministic labels) + tail-focused sampling from
existing IQA datasets. Does NOT require VLM pseudo-labels for the synthetic component.

### Gap 2: Form and Structured Layout Diversity (HIGH)

**The problem:** Zero form/tabular/invoice coverage. Forms have fundamentally different
quality characteristics — structural alignment matters more than text sharpness.

**Evidence:** Form layout SRCC = 0.33 (VLMs), -0.09 (SigLIP2). VLMs assess readability,
not structural integrity.

**Impact:** Enterprise document processing (insurance claims, tax forms, government
applications) gets unreliable quality scores. Market-limiting gap.

**Strategy:** VLM pseudo-labels are unreliable here. Requires either (a) human annotation
of 200-300 forms with structure-aware rubric, or (b) structural quality heuristics
(line detection, field alignment, OCR confidence per field). **Deferred to post-Tier 2.**

### Gap 3: Script Homogeneity (HIGH)

**The problem:** 75% Hans, 23% Latin. Zero Arabic, Devanagari, Thai, Korean, Cyrillic,
Tibetan coverage. Global deployment impossible.

**Evidence:** Script OOD SRCCs are actually the best category (0.73-0.85), suggesting
SigLIP2's vision features are partially script-invariant. But the gap is real for
fine-grained quality assessment.

**Impact:** Cannot deploy in Middle East, South Asia, Southeast Asia, or Eastern Europe.

**Strategy:** Two complementary approaches:
(a) **Synth-multiscript-v3 degradation replay**: Select pristine base images across 27
scripts, apply controlled degradation at known parameters → deterministic IQA labels at
zero VLM cost. This is the primary source for script expansion.
(b) **VLM pseudo-labels on real multi-script datasets**: mdiw13 (290K, 13 scripts),
kuzushiji, Arabic-docs — VLM Tier A reliability (SRCC 0.6-0.85).
(c) **Thousand Character Classic** (391 images): Historical CJK calligraphy with
legibility-to-quality mapping across 6 script styles — fills CJK vertical text gap.
Requires per-script calibration validation for VLM-labeled sources.

### Gap 4: Capture Method Bias (HIGH)

**The problem:** 91% synthetic, 9% camera. No scanner diversity. Mobile capture introduces
perspective distortion, shadows, motion blur — none represented in training.

**Evidence:** DPI extreme SRCC is negative (-0.15 to -0.41), suggesting the model confuses
resolution with quality. Camera-captured documents have fundamentally different resolution
characteristics.

**Impact:** Mobile-first deployments (field agents, self-service kiosks) get inverted
quality scores.

**Strategy:** Source from SmartDoc-QA (4.3K mobile), RealDAE (1.2K paired camera/flatbed),
sd7k (7K shadows), midv500 (3.6K mobile ID). VLM pseudo-labels are marginal for
camera-captured documents (readability-vs-optical-quality gap). Human annotation of
300-500 camera samples recommended.

### Gap 5: Degradation Mode Coverage (MEDIUM)

**The problem:** Training synthetic degradation is controlled but narrow. Real documents
have compound degradations (blur + noise + compression + skew simultaneously). Models
trained on single-condition degradation fail 15-25% worse on compound inputs.

**Evidence:** Heavily degraded SRCC = 0.24 (VLMs), -0.05 (SigLIP2). VLMs hit a floor
effect where everything looks "bad."

**Strategy:** The synth-multiscript-v3 hybrid augmentation pipeline (Augraphy + Albumentations)
already implements compound degradation with reproducible parameter replay via stored
degradation seeds. Apply to both DIQA-5000 clean samples AND diverse synth-multiscript-v3
base images. Labels derived from degradation parameters — no VLM needed. The v4 font
diversity strategy adds adversarial fonts (5% tier) that create naturally harder quality
assessment targets, ensuring the model learns quality from degradation signals rather
than font familiarity.

### Gap 6: Processing-State Blindness (MEDIUM, DEFERRED)

**The problem:** Binarized documents, DPI extremes, and pristine documents are universal
failure modes where NO current model (VLM or fine-tuned) produces reliable quality
estimates.

**Evidence:** Binarized SRCC = -0.37 (all models). The quality concept itself differs:
binarization quality depends on character completeness and stroke preservation, not
blur or color.

**Strategy:** Cannot be addressed with current pseudo-label pipeline. Requires:
(a) domain-specific quality metrics (DIBCO metrics for binarized, DPI-normalized
assessment), (b) human annotation with processing-state-aware rubric, or (c) future
VLM improvements. **Excluded from this expansion plan.**

---

## 3. Structural Risks and Mitigations

### 3.1 The Circular Training Trap

**Risk:** SigLIP2 embeddings define the OOD detection space (Mahalanobis distance). After
retraining on OOD data, the embedding distribution shifts. The OOD detector fitted on
DIQA-5000 becomes unreliable — documents that were OOD may no longer be flagged, and
vice versa.

**Mitigations:**

1. **Re-fit OOD detector after each tier.** Extract new embeddings, recompute mean and
   precision matrix. Never carry forward a stale OOD detector.
2. **Freeze vision encoder during initial expansion.** Train only the quality head in
   Tiers 1-2. This preserves embedding space stability. Full fine-tune only in Tier 3+.
3. **Never use SigLIP2 self-labels.** Quality scores always come from VLM consensus or
   known degradation formulas. SigLIP2 is used only for OOD detection and uncertainty,
   never for score generation.
4. **External validation at every gate.** DIQA-5000 val (500 images with human GT) is
   the regression anchor. If ID performance drops, rollback.

### 3.2 VLM Calibration Transfer

**Risk:** Isotonic calibration fitted on DIQA-5000 may not transfer to OOD categories.
VLMs may under-rate unfamiliar scripts (perceived as lower quality) or over-rate pristine
born-digital documents.

**Mitigations:**

1. **Per-category bias monitoring.** Track mean predicted MOS per source dataset. If mean
   shifts >0.5 MOS from expected range, flag for investigation.
2. **Bridge set validation.** Before scaling any category, run VLM consensus on 50-100
   images and compare inter-model agreement. If JSD > 0.15 across models, that category
   needs separate calibration or human labels.
3. **Conservative weight floor.** Start all OOD pseudo-labels at weight 0.5 (not 1.0).
   Increase only after validation confirms quality.

### 3.3 Pseudo-Label Quality Ceiling

**Risk:** SigLIP2 trained on VLM pseudo-labels cannot exceed VLM quality. Best VLM wSRCC
is 0.708 (Gemini 3 Flash on DIQA-5000). Two-model consensus is 0.778 on synthetic OOD.
This is the ceiling for pseudo-labeled categories.

**Mitigations:**

1. **Use pseudo-labels for domain coverage, not quality ceiling.** The goal is to prevent
   catastrophic OOD failure (SRCC going negative), not to achieve 0.9+ on every category.
   Even noisy pseudo-labels at SRCC 0.6 are dramatically better than SRCC -0.37.
2. **Prioritize deterministic labels.** Controlled degradation provides exact labels with
   no noise. Use this for quality-tail correction (Gap 1) and degradation coverage (Gap 5).
3. **Track the ceiling.** After each tier, compare model SRCC on holdout against VLM SRCC
   on the same holdout. If the model approaches VLM performance, further pseudo-label
   expansion yields diminishing returns.

### 3.4 Confident Wrong Answers

**Risk:** When both VLMs are systematically wrong in the same direction (e.g., both rate
binarized documents as "poor" regardless of actual quality), they agree (low JSD) and
the pseudo-label passes uncertainty filtering with high confidence. The pipeline cannot
detect confident agreement on wrong answers.

**Mitigations:**

1. **Category-level exclusion, not sample-level filtering.** For Tier C categories
   (binarized, pristine, DPI extremes), exclude the entire category rather than relying
   on per-sample uncertainty to catch errors.
2. **Human bridge sets for Tier B categories.** For form layouts and heavily degraded
   documents, annotate 50-100 human labels. Measure VLM consensus vs. human SRCC. If
   VLM SRCC < 0.5 on the bridge set, exclude the category.

### 3.5 OCR-Based Quality Validation

**Opportunity:** Datasets with ground-truth text transcriptions enable an independent
quality validation signal via OCR engine accuracy. This is a powerful cross-check
because OCR performance is causally related to document image quality — degraded images
produce worse OCR — and is entirely independent of both VLM quality judgments and
SigLIP2 predictions.

**Strategy:**

1. **Prioritize samples with text GT.** When selecting images from source datasets,
   prefer images that include text transcription ground truth. This enables OCR-based
   quality validation at zero additional annotation cost.
2. **OCR quality correlation.** Run an OCR engine (e.g., Tesseract, PaddleOCR) on
   selected images. Compute character error rate (CER) or word error rate (WER) against
   text GT. CER/WER should correlate negatively with assigned quality scores —
   higher quality images should produce lower error rates.
3. **Disagreement flagging.** If VLM pseudo-label says "excellent" but OCR CER > 20%,
   or VLM says "bad" but OCR CER < 5%, flag for review. These disagreements indicate
   either VLM miscalibration or unusual document characteristics (e.g., adversarial
   fonts that are legible but VLMs rate as low quality).
4. **Per-script OCR calibration.** OCR accuracy varies by script (Latin engines are
   strongest; CJK is moderate; Tibetan/Myanmar are weak). Calibrate CER thresholds
   per script family before using OCR as a quality gate.

**Datasets with text GT (prioritize):**

- **OHR-Bench** (Tier 1): Full text transcription GT
- **Tobacco800** (Tier 1): Document text content available
- **SmartDoc-QA** (Tier 1): QA pairs with source text
- **Kuzushiji** (Tier 2): Character-level annotations
- **Thousand Character Classic** (Tier 2): Known fixed literary text — unique advantage
  of exact text content without OCR dependency
- **mdiw13** (Tier 2): Partial text annotations per script
- **Synth-multiscript-v3** (Tiers 1-2): Full text GT by construction (synthetic)
- **DocLayNet** (Tier 2): Layout annotations with text regions

**Non-text-GT sources (lower priority for validation, still usable):**

- **RealDAE** (Tier 1): Paired images but no text GT
- **sd7k** (Tier 3): Shadow images, no text GT
- **warpdoc** (Tier 3): Geometric distortion, no text GT
- **midv500** (Tier 4): ID documents, privacy-restricted text

---

## 4. Staged Expansion Plan

### Tier 0: Validation Infrastructure (Week 0-1)

**Goal:** Establish measurement framework before adding any training data.

**Deliverables:**

| Eval Set | Size | Source | Purpose |
| ---------- | ------ | -------- | --------- |
| DIQA-5000 val | 500 | Existing (GT labels) | ID regression anchor |
| OOD doc-type eval | 500 | RVL-CDIP (stratified by 16 types) | Layout/domain diversity |
| OOD degradation eval | 200 | Tobacco800 (archival) | Degradation robustness |
| OOD capture eval | 100 | SmartDoc-QA (mobile) | Capture method diversity |
| OOD script eval | 200 | mdiw13 (50/script x 4 scripts) | Script generalization |
| **Total eval** | **1,500** | | **Never trained on** |

**All OOD eval sets labeled by 2-model VLM consensus** (Gemini 3.1 Flash Lite + Qwen 3.5-122B).
Cost: 1,000 images x $0.002 = $2.00.

**Baseline metrics:** Run current SigLIP2-IQA on all eval sets. Record wSRCC, SRCC per
dimension, MAE, PLCC. These are the regression benchmarks for all future tiers.

**Validation gate:** VLM consensus wSRCC >= 0.70 on DIQA-5000 val.
If FAIL: iterate on VLM prompt optimization before proceeding.

---

### Tier 1: Quality Balance + Domain Seed (Week 2-4)

**Goal:** Fix the 61% fair / 0.5% excellent distribution skew and add initial domain
diversity. Highest-ROI intervention because the model has almost no training signal
at quality extremes.

**Data composition:**

| Source | Samples | Target Quality Range | Label Strategy |
| -------- | --------- | --------------------- | ---------------- |
| **Controlled degradation** (DIQA-5000 base) | 1,400 | Bad/poor/degraded (from 350 train images) | Deterministic: original MOS - degradation offset |
| **Synth-multiscript-v3 degradation replay** | 2,000 | Full range across 10+ scripts | Deterministic: degradation params → IQA labels |
| **OHR-Bench** | 1,200 | Stratified: 400 excellent, 400 poor, 400 bad | VLM consensus (OHR quality scores as sampling prior) |
| **RealDAE** | 600 | 300 camera (degraded), 300 flatbed (pristine) | VLM consensus |
| **Tobacco800** | 400 | Archival — mostly poor/bad | VLM consensus |
| **SmartDoc-QA** | 500 | 250 high, 250 low quality | VLM consensus |
| **Total** | **6,100** | | |

**Controlled degradation from DIQA-5000** (1,400 samples, no VLM cost):

- Take all 350 DIQA-5000 base train images with GT labels
- Apply 4 degradation profiles at controlled severity, producing 4 variants each
- Degradation types: JPEG (q=10,20,40,70), blur (sigma=1,2,4,8),
  noise (sigma=5,15,30,50), downscale-upscale (0.25x, 0.5x, 0.75x, 0.9x)
- Labels: original GT MOS - calibrated degradation offset (validated against
  VLM consensus on 100-image pilot)
- Soft labels: use Gaussian CDF with sigma proportional to degradation severity

**Synth-multiscript-v3 degradation replay** (2,000 samples, no VLM cost):

- Select 200 pristine base images from synth-multiscript-v3, stratified across
  10+ scripts (prioritize underrepresented: Tibt, Mymr, Ethi, Deva, Thai, Hebr)
- Apply the existing hybrid augmentation pipeline (Augraphy + Albumentations) at
  5 quality tiers: PRISTINE, HIGH, MEDIUM, LOW, DEGRADED
- Include compound degradations: paper aging + ink degradation + blur combinations
- Labels derived from degradation parameters using existing formula:
  `blur_severity = sigma/5.0`, `noise_severity = amount/0.05`,
  `overall_quality = 1.0 - max(component_severities)`
- Map the 0-1 overall_quality to DIQA's 1-5 MOS scale: `MOS = 1.0 + 4.0 * quality`
- v4 font diversity: apply tiered font sampling (SYSTEM 40%, REGIONAL 25%,
  STYLISTIC 15%, HANDWRITING 15%, ADVERSARIAL 5%) to prevent font overfitting
- Stored degradation seeds enable bit-for-bit reproducibility

**VLM labeling** (2,700 samples):

- Two-model consensus: Gemini 3.1 Flash Lite + Qwen 3.5-122B
- Agreement |delta| <= 0.5 MOS: use mean (expected ~70%)
- Disagreement 0.5 < |delta| <= 1.0: use mean, set std=1.2 (elevated uncertainty)
- Disagreement |delta| > 1.0: exclude (expected ~5-10%)
- Cost: 2,700 x $0.002 = $5.40

**Holdout:** 610 images (10%), stratified across sources. Never trained on.

**Training configuration:**

- 5,490 new + 3,500 DIQA-5000 = **8,990 total**
- DIQA-5000 at weight 1.0, pseudo-labels at weight 0.5
- Deterministic degradation labels (controlled + synth-multiscript-v3) at weight 0.7
- Phase 1: 10 epochs head-only warmup (freeze vision encoder)
- Phase 2: 40 epochs full fine-tune at reduced LR (1e-6)
- Compute: ~2-3 days on 4x A100

**Validation gate (ALL must pass):**

| Criterion | Threshold | Rollback Action |
| ----------- | ----------- | ----------------- |
| ID no-regression | DIQA-5000 val wSRCC >= baseline - 0.01 | Reduce pseudo_label_weight to 0.3, ablate by source |
| OOD improvement | Tier 0 OOD eval wSRCC >= baseline + 0.02 | Partial pass if ID holds; reduce Tier 2 scope |
| Quality tail | SRCC on excellent+bad holdout subset >= 0.50 | Re-examine degradation calibration; remove VLM-labeled tails |
| Distribution check | All 5 quality levels have >2% predicted mass on val | Rebalance sampling, increase tail representation |

---

### Tier 2: Script and Layout Diversity (Week 5-8)

**Goal:** Break the 75% Hans / 23% Latin distribution. Add layout diversity from
born-digital sources where VLM quality assessment is most reliable.

**Data composition:**

| Source | Samples | Script/Layout | Label Strategy |
| -------- | --------- | --------------- | ---------------- |
| **Synth-multiscript-v3 (new scripts)** | 2,000 | 15+ scripts not in Tier 1 (Cyrl, Grek, Armn, Geor, Khmr, etc.) | Deterministic: degradation params |
| **mdiw13** | 1,500 | 13 scripts (115/script) — real document diversity | VLM consensus (Tier A) |
| **DocLayNet** | 1,500 | 11 layout classes (136/class) | VLM consensus (Tier A) |
| **Kuzushiji** | 500 | Historical Japanese cursive | VLM consensus (Tier A) |
| **Thousand Character Classic** | 300 | Historical CJK calligraphy, 6 script styles | VLM consensus + legibility prior |
| **RVL-CDIP** | 800 | 16 doc types (50/type) | VLM consensus (Tier A) |
| **Total** | **6,600** | | |

**Script sampling strategy:**

- Target: no single script >30% of new data
- Minimum 100 images per script for VLM-labeled sources
- Synth-multiscript-v3 provides deterministic labels for 15+ additional scripts at zero
  VLM cost — use for scripts where real data is scarce (Geor, Khmr, Sinh, Orya, etc.)
- Include historical/degraded variants (Kuzushiji, Thousand Character Classic) alongside
  clean synthetic for temporal diversity
- Per-script calibration: run VLM consensus on 50 images/script, compare inter-model
  agreement. If JSD > 0.15 for a script, cap that script's weight at 0.3.

**Excluded from Tier 2:**

- Binarized images (VLM SRCC negative — would corrupt training)
- DPI extremes (<72 or >600 DPI — VLM anti-correlated)
- Pristine born-digital with no defects (VLM cannot discriminate)

**VLM labeling:** Same two-model consensus protocol as Tier 1. Applied only to real
document sources (mdiw13, DocLayNet, Kuzushiji, Thousand Character Classic, RVL-CDIP).
Synth-multiscript-v3 uses deterministic degradation-parameter labels.
Cost: 4,600 x $0.002 = $9.20

**Holdout:** 660 images (10%), stratified by script. Becomes permanent script-diversity
OOD eval set.

**Training configuration:**

- 5,940 new + 8,990 Tier 1 = **14,930 total**
- DIQA-5000 upsampled 2x to prevent catastrophic forgetting
- Synth-multiscript-v3 deterministic labels at weight 0.7
- VLM pseudo-labels at weight adjusted based on Tier 1 validation findings
- Keep vision encoder frozen for script expansion (preserve embedding stability)
- Compute: ~3 days on 4x A100

**Post-training:** Re-fit OOD detector on new embedding distribution. Extract embeddings
from full training set, recompute Mahalanobis mean + precision matrix.

**Validation gate:**

| Criterion | Threshold | Rollback Action |
| ----------- | ----------- | ----------------- |
| ID no-regression | DIQA-5000 val wSRCC >= Tier 1 - 0.01 | Increase DIQA upsampling to 3x, reduce expansion |
| Script OOD | Script eval wSRCC >= Tier 1 + 0.03 | Per-script ablation: remove non-improving scripts |
| Layout OOD | Doc-type eval wSRCC >= Tier 1 + 0.02 | Acceptable if script metrics pass |
| Cross-script consistency | SRCC std across scripts <= 0.15 | Upsample failing scripts, downsample dominant |

---

### Tier 3: Degradation Hardening + Camera Capture (Week 9-12)

**Goal:** Fill degradation mode gaps (compound degradation, shadows, warping) and add
real camera-captured diversity. This tier has the highest label-quality risk due to
marginal VLM reliability on degraded documents.

**Data composition:**

| Source | Samples | Focus | Label Strategy |
| -------- | --------- | ------- | ---------------- |
| **Compound degradation** (synthetic) | 2,000 | Blur+noise, blur+JPEG, shadow+blur, noise+contrast+JPEG | Deterministic (degradation params) |
| **sd7k** | 1,000 | Document shadows | VLM consensus (3-model for disagreements) |
| **warpdoc** | 500 | Geometric warping | VLM consensus (strict: \|delta\| <= 0.3) |
| **SmartDoc-QA extended** | 800 | Mobile capture beyond Tier 1 | VLM consensus |
| **midv500** | 500 | Mobile ID document capture | VLM consensus |
| **AnyPhotoDoc** | 500 | Camera dewarping pairs | VLM consensus |
| **Total** | **5,300** | | |

**Compound degradation pipeline** (2,000 samples, deterministic labels):

| Combination | Count | Pipeline | Label |
| ------------- | ------- | ---------- | ------- |
| Blur + JPEG compression | 500 | Gaussian/motion blur -> JPEG Q=30-50 | MOS = base - blur_offset - jpeg_offset |
| Blur + noise | 400 | Gaussian blur -> additive noise | MOS = base - max(blur_offset, noise_offset) - 0.3 |
| Shadow + blur (camera domain) | 400 | Shadow overlay -> motion blur | MOS = base - shadow_offset - blur_offset |
| Noise + contrast + JPEG | 400 | Noise -> contrast reduction -> JPEG | MOS = base - compound_offset |
| Blur + skew + noise (three-way) | 300 | Rotation -> blur -> noise | MOS = base - compound_offset |

Calibration: validate degradation-to-MOS curves against VLM consensus on a 200-image
pilot before generating the full 2,000.

**Three-model tiebreaker for degraded documents:**
When Flash Lite + Qwen 122B disagree by >0.5 MOS on shadow/warping images, add GPT-4.1
as third annotator. Majority vote on quality level. Cost overhead: ~$4 for ~400 tiebreaker
calls.

**Holdout:** 530 images (10%), stratified by degradation type.

**Training configuration:**

- 4,770 new + 14,930 Tier 2 = **19,700 total**
- DIQA-5000 upsampled 3x (anchor weight increases as dataset grows)
- Synthetic compound samples at weight 0.7 (deterministic labels)
- VLM pseudo-labels at weight 0.5 (camera/shadow/warp)
- **Full fine-tune allowed at this tier** (vision encoder unfrozen, LR=5e-7)
- Compute: ~3-4 days on 4x A100

**Post-training:** Re-fit OOD detector (critical after unfreezing vision encoder).

**Validation gate:**

| Criterion | Threshold | Rollback Action |
| ----------- | ----------- | ----------------- |
| ID no-regression | DIQA-5000 val wSRCC >= Tier 2 - 0.01 | Reduce degradation weight to 0.3, re-freeze encoder |
| Degradation robustness | Degradation holdout MAE <= 0.40 MOS | Recalibrate degradation formulas on VLM pilot |
| Shadow/warp SRCC | Shadow+warp holdout SRCC >= 0.50 | Accept if MAE <= 0.50; these are known hard cases |
| Compound check | Compound holdout SRCC >= 0.60 | Check degradation curves; reduce compound severity range |

---

### Tier 4: Active Learning Refinement (Week 13-16)

**Goal:** Use the Tier 3 model's uncertainty estimates to find remaining failure cases
and surgically close gaps. This is where the OOD detector and confidence weighting
pipeline pay off.

**Approach:**

1. **Mine hard examples.** Run Tier 3 model + OOD detector on all unused images from
   Tier 1-3 source pools (~50K+ available from mdiw13, DocLayNet, kuzushiji, OHR-Bench).
   Rank by uncertainty: high Mahalanobis distance, high predicted variance, high VLM
   disagreement.

2. **Sample 3,000 highest-uncertainty images:**
   - ~1,000 from rare scripts (high Mahalanobis distance, boundary cases)
   - ~1,000 from unusual layouts/quality extremes (high predicted variance)
   - ~1,000 from VLM-disagreement cases (hardest to label)

3. **Labeling:**
   - 2,000 high-uncertainty but VLM-agreeable: standard two-model consensus
   - 1,000 VLM-disagreement cases: three-model consensus. If all three disagree
     by >0.5 MOS pairwise, exclude (genuinely ambiguous).

4. **Optional human annotation checkpoint** (strongly recommended):
   - Take the 200 highest-uncertainty images that passed VLM consensus
   - Get 3 human ratings each (~$300 at $0.50/image/rater)
   - Validate VLM consensus vs. human SRCC on hard cases
   - If VLM SRCC < 0.5 on bridge set, cap all Tier 4 weights at 0.3

**Cost:** 2,000 x $0.002 + 1,000 x $0.005 + optional $300 human = ~$15-315

**Holdout:** 300 images (10%), all from uncertainty-mined pool.

**Training configuration:**

- 2,700 new + 19,700 Tier 3 = **22,400 total**
- DIQA-5000 upsampled 4x
- Uncertainty-mined samples at variable weight based on confidence_weight
  (sigma_sq + entropy weighting from existing pipeline)
- Compute: ~3-4 days on 4x A100

**Post-training:** Final OOD detector re-fit. Measure OOD coverage improvement.

**Validation gate (tighter tolerances):**

| Criterion | Threshold | Action |
| ----------- | ----------- | -------- |
| ID no-regression | DIQA-5000 val wSRCC >= Tier 3 - 0.005 | Rollback; filter to VLM-agreement-only |
| Full OOD | Combined OOD eval wSRCC >= Tier 3 + 0.01 | Accept Tier 3 as final (diminishing returns) |
| Calibration | ECE on Tier 4 holdout <= 0.10 | Temperature scaling on val set |
| OOD detector | FPR at 95% TPR <= 10% (from 14.6%) | Accept improvement; re-evaluate thresholds |

---

## 5. Cumulative Budget

| Tier | New Samples | Cumulative Train | VLM Cost | Compute | Key Metric Target |
| ---- | ----------- | ---------------- | -------- | ------- | ----------------- |
| 0 | 0 (1,500 eval) | 3,500 | $2.00 | 1 day | Measurement framework |
| 1 | 5,490 | 8,990 | $5.40 | 2-3 days | OOD wSRCC +0.02 |
| 2 | 5,940 | 14,930 | $9.20 | 3 days | Script OOD wSRCC +0.03 |
| 3 | 4,770 | 19,700 | $10.60 | 3-4 days | Degradation SRCC >= 0.50 |
| 4 | 2,700 | 22,400 | $13-313 | 3-4 days | ECE <= 0.10, no ID regression |
| **Total** | **18,900** | **22,400** | **~$40-340** | **~12-15 days** | |

### Holdout Budget (Never Trained On)

| Set | Size | Created At | Purpose |
| --- | ---- | ---------- | ------- |
| DIQA-5000 val | 500 | Pre-existing | ID regression anchor (human GT) |
| Tier 0 OOD eval | 1,000 | Tier 0 | Cross-domain baseline |
| Tier 1 holdout | 610 | Tier 1 | Quality-tail validation |
| Tier 2 holdout | 660 | Tier 2 | Script diversity validation |
| Tier 3 holdout | 530 | Tier 3 | Degradation validation |
| Tier 4 holdout | 300 | Tier 4 | Uncertainty calibration |
| **Total holdout** | **3,600** | | **Never enters training** |

---

## 6. What This Plan Does NOT Address

These gaps are acknowledged but deferred due to VLM unreliability or insufficient tooling:

### 6.1 Binarized Documents (VLM SRCC = -0.37)

- **Why deferred:** All models (VLM and fine-tuned) produce anti-correlated quality
  scores. The quality concept for binarized documents (character completeness, stroke
  preservation) is orthogonal to the visual quality that current models assess.
- **Future approach:** DIBCO-style binarization quality metrics, human annotation with
  binarization-specific rubric (200+ samples), or specialized binarization quality model.

### 6.2 DPI-Extreme Documents (VLM SRCC = -0.15 to -0.41)

- **Why deferred:** VLMs confuse resolution with quality. High DPI = "good" regardless
  of content quality; low DPI = "bad" regardless of legibility at intended scale.
- **Future approach:** DPI-normalized quality assessment. Condition on resolution
  metadata or learn DPI-invariant features via multi-resolution training.

### 6.3 Pristine Documents (VLM SRCC = -0.09)

- **Why deferred:** No quality signal — all models output near-constant scores. The
  quality variation within pristine documents (sub-pixel rendering, anti-aliasing)
  is below VLM perceptual threshold.
- **Future approach:** Determine if fine-grained pristine quality is even needed.
  If all pristine documents should map to "excellent," a simple heuristic (low OOD
  distance + low degradation score -> excellent) may suffice without training data.

### 6.4 Form-Specific Quality (VLM SRCC = 0.17-0.33)

- **Why partially deferred:** VLMs assess readability, not structural integrity.
  Form quality depends on field alignment, checkbox legibility, table border
  completeness. DocLayNet form samples in Tier 2 provide layout awareness but not
  form-specific quality assessment.
- **Future approach:** Structure-aware quality rubric + human annotation (200-300 forms),
  or rule-based quality features (line detection, field alignment metrics).

### 6.5 Human Annotation at Scale

- **Why deferred:** The DIQA-5000 protocol ($50K+ for 5,000 images, 15 raters) is
  too expensive to repeat. This plan relies on VLM pseudo-labels and controlled
  degradation instead.
- **When needed:** After Tier 4, if the model saturates at the VLM quality ceiling
  (~0.78 wSRCC on OOD), further improvement requires human labels. Targeted annotation
  of 500-1,000 images across failure categories ($5-10K) would break the ceiling.

---

## 7. Decision Framework

### When to Stop Expanding

Stop dataset expansion when ANY of these conditions is met:

1. **Diminishing returns:** Tier N+1 OOD improvement < 0.005 wSRCC over Tier N
2. **VLM ceiling reached:** Model SRCC on OOD holdout approaches VLM consensus SRCC
   (within 0.02), meaning further pseudo-labels cannot improve performance
3. **ID regression:** Any tier causes >0.02 wSRCC regression on DIQA-5000 val that
   cannot be recovered by weight adjustment
4. **Category saturation:** Per-category SRCC on all Tier A categories exceeds 0.70

### When to Invest in Human Annotation

Trigger human annotation when:

1. **Category-specific VLM SRCC < 0.5** on bridge set (50-100 images with human labels)
2. **Inter-model JSD > 0.15** for a target category (VLMs fundamentally disagree)
3. **Tier B category is business-critical** (e.g., forms for enterprise deployment)
4. **Model saturates** at VLM ceiling after Tier 4

### Emergency Rollback Protocol

If any tier corrupts the model:

1. Revert to previous tier's checkpoint
2. Identify contaminating source via per-dataset ablation (remove one source at a time)
3. Reduce pseudo_label_weight to 0.1 and re-run
4. If still failing, exclude the entire tier's data and proceed to next tier
5. Document failure mode for future investigation

---

## 8. Relationship to Existing Infrastructure

### Pseudo-Label Pipeline (`src/uncertainty/`)

This plan uses the existing pipeline components:

| Component | Role in Expansion | Modifications Needed |
| ----------- | ------------------ | --------------------- |
| `ood_wrapper.py` | Tier 4 uncertainty mining | Re-fit after each tier |
| `fusion.py` | Acceptance decisions for VLM labels | None — use existing tiers |
| `vlm_validator.py` | Tier 2 validation (budget-capped) | Configure for 3-model tiebreaker |
| `pseudo_label.py` | Orchestrate labeling pipeline | Add per-source weight config |
| `confidence_weight.py` | Training sample weights | Set floor at 0.3 for pseudo-labels |
| `gaussian_to_discrete.py` | Convert MOS to soft labels | None |
| `cross_validator.py` | SigLIP2 vs DeQA agreement | Used in uncertainty mining |

### New Infrastructure Required

1. **Controlled degradation pipeline:** Augraphy + Albumentations wrapper that takes
   a clean image + GT MOS and produces degraded variants with deterministic labels.
   Partially exists in `research/ocr_iqa_correlation/distortion/apply_distortions.py`.

2. **Per-source weight configuration:** Training script needs to support per-dataset
   weight multipliers (DIQA=1.0, pseudo-label=0.5, synthetic=0.7).

3. **Tier-specific training manifests:** JSON files that specify source, weight, and
   holdout membership for each image.

4. **OOD detector re-fitting script:** Automated re-extraction + re-fit after each
   training tier.

---

## Appendix A: Category-Level Pseudo-Label Feasibility

Critical assessment from reliability analysis agent. Confidence values reflect
uncertainty about real-world performance (synthetic SRCC numbers are upper bounds).

| Category | VLM SRCC (Synthetic) | Feasibility | Confidence | Est. >1 MOS Error Rate | Recommended Strategy |
| ---------- | --------------------- | ------------- | ------------ | ---------------------- | --------------------- |
| Fraktur | 0.76-0.77 | Moderate-High | 60% | 15-25% | Standard pseudo-label |
| Nastaliq | 0.77-0.85 | Moderate-High | 55% | 10-20% | Standard pseudo-label |
| Tibetan | 0.73-0.80 | Moderate-High | 55% | 15-25% | Pseudo-label + bridge validation |
| Myanmar | 0.76-0.76 | Moderate | 50% | 15-25% | Pseudo-label + bridge validation |
| Ethiopic | 0.77-0.80 | Moderate | 50% | 15-25% | Standard pseudo-label |
| CJK Vertical | 0.62-0.75 | Low-Moderate | 35% | 25-35% | Pseudo-label at reduced weight |
| Multiscript | 0.66-0.76 | Low-Moderate | 30% | 25-40% | Pseudo-label at reduced weight |
| Form Layouts | 0.17-0.33 | Very Low | 15% | 50-70% | Defer; needs human annotation |
| Heavily Degraded | 0.17-0.24 | Very Low | 10% | 50-70% | Use synthetic degradation instead |
| Binarized | -0.34 to -0.49 | None | 5% | 60-80% | **Exclude entirely** |
| Pristine | -0.09 to +0.30 | None | 10% | 40-60% | Heuristic mapping |
| Very High DPI | -0.15 to -0.11 | None | 10% | 50-70% | **Exclude entirely** |
| Very Low DPI | -0.41 to -0.22 | None | 5% | 60-80% | **Exclude entirely** |

> **Key caveat:** All SRCC values are from synthetic OOD data with controlled GT derived
> from generation parameters. Real-world reliability is unknown and likely lower. The
> "Confidence" column reflects this uncertainty. Bridge set validation (50-100 human-labeled
> images per category) is the only way to resolve this uncertainty before committing to
> large-scale pseudo-labeling.

---

## Appendix B: Interaction with Image Detection Project

The image detection project (`/home/byron/dev/image_detection/`) has its own 14-dimension
diversity requirements for SigLIP2 multi-task training. While this DIQA expansion plan
serves a narrower goal (quality assessment only), several synergies exist:

1. **Shared source datasets:** mdiw13, DocLayNet, OHR-Bench, SmartDoc-QA appear in both
   projects. Coordinate sampling to avoid duplicate processing.

2. **SigLIP2 backbone:** Both projects use SigLIP2-Base. The image detection project's
   multi-task training may produce a backbone with better OOD features than DIQA-only
   fine-tuning. Consider using the multi-task backbone as initialization for DIQA expansion.

3. **Degradation pipeline:** The image detection project's IQA Synthetic dataset (100K
   planned) uses the same Augraphy + Albumentations pipeline. Share degradation profiles
   and quality calibration curves.

4. **Script detection:** The image detection project's script detection head (108 classes)
   can provide script metadata for DIQA expansion images, enabling per-script calibration
   validation without manual annotation.
