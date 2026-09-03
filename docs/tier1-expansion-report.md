# DIQA-5000_1 Tier 1 Expansion Report

> **Version**: 1.0.0
> **Date**: 2026-03-15
> **Dataset**: DIQA-5000_1 (13,163 training samples)
> **Parent**: DIQA-5000_0 (3,500 training samples)

## Executive Summary

DIQA-5000_1 expands the baseline training set from 3,500 to 13,163 samples (+276%) through three complementary data streams. The expansion addresses critical weaknesses in the baseline: narrow quality distribution (concentrated in fair/good), limited capture diversity (91% synthetic), single script family (75% Hans, 23% Latn), and zero representation of real-world document degradation.

The expansion adds deterministic labels from controlled degradation (Stream 1) and fresh synthetic generation (Stream 2), plus VLM consensus pseudo-labels for real document images (Stream 3). All 5 quality levels now exceed the 2% minimum mass threshold, with the combined distribution spanning MOS 0.63-4.88.

---

## 1. Training Data Composition

### 1.1 Stream Overview

| Stream | Samples | % of Total | Real | Synthetic | Label Method | Weight | Purpose |
|--------|--------:|------------|-----:|----------:|-------------|--------|---------|
| **Base** (DIQA-5000_0) | 3,500 | 26.6% | 350 | 3,150 | Human MOS | 1.0 | Anchor: human ground truth |
| **Stream 1** (Controlled Degradation) | 1,260 | 9.6% | 0 | 1,260 | Deterministic | 0.7 | Quality balance: fill poor/bad bins |
| **Stream 2** (Synth Multiscript) | 1,800 | 13.7% | 0 | 1,800 | Deterministic | 0.7 | Script diversity: 24 ISO 15924 scripts |
| **Stream 3** (VLM Consensus) | 6,603 | 50.2% | 6,603 | 0 | VLM pseudo-label | 0.5 | Domain diversity: real documents |
| **Total Training** | **13,163** | 100% | **6,953** | **6,210** | | | |
| **Holdout** | 1,072 | — | | | | | 10% stratified by source |

**Real-world document ratio: 10.0% (baseline) → 52.8% (Tier 1) by count.** The 350 real images in the baseline are the original DIQA source documents (before synthetic quality variants were applied); the remaining 3,150 are synthetic degradation variants of those same 350 documents. Stream 3 adds 6,603 genuinely distinct real documents from 7 external datasets.

However, Stream 3 carries the lowest confidence weight (0.5) because all labels are VLM pseudo-labels rather than human annotations or deterministic computations. When accounting for training weights, the **effective contribution** shifts:

| Stream | Samples | Weight | Effective Weight | Eff. % |
|--------|--------:|-------:|-----------------:|-------:|
| Base | 3,500 | 1.0 | 3,500 | 36.0% |
| Stream 1 | 1,260 | 0.7 | 882 | 9.1% |
| Stream 2 | 1,800 | 0.7 | 1,260 | 13.0% |
| Stream 3 | 6,603 | 0.5 | 3,302 | 34.0% |
| **Total** | **13,163** | | **8,944** | **100%** |

By effective weight, synthetic sources (base + S1 + S2) contribute 58.0% and real-world sources (Stream 3) contribute 34.0%, with the human-annotated base remaining the single largest influence at 36.0%. This ensures the model is grounded in human judgments while still benefiting from the diversity that real-world documents provide.

### 1.2 Source Dataset Breakdown

| Source | Samples | % | Capture | Text GT | Notes |
|--------|--------:|---:|---------|---------|-------|
| diqa_5000_human_gt | 3,500 | 26.6% | Synthetic + camera | No | Human MOS annotations |
| smartdoc_qa | 3,834 | 29.1% | Camera (robotic arm) | Yes | 21 train docs, 2 phones |
| synth_fresh_generation | 1,800 | 13.7% | Synthetic | No | 24 scripts, fixed font diversity |
| diqa_degradation | 1,260 | 9.6% | Degraded DIQA | No | 4 degradation tiers |
| funsd_plus | 924 | 7.0% | Scanner | Yes | Form documents, train split |
| tobacco800 | 630 | 4.8% | Scanner (archival) | No | Real aging and degradation |
| realdae | 450 | 3.4% | Camera | No | Camera vs flatbed pairs |
| ocr_quality | 360 | 2.7% | Mixed | Yes | Multilingual (55% Hans, 40% Latn) |
| sroie | 270 | 2.1% | Camera | Yes | Retail receipts |
| funsd | 135 | 1.0% | Scanner | Yes | Form documents, train split |

---

## 2. Quality Distribution Analysis

### 2.1 Level Mass Distribution

The baseline DIQA-5000_0 has a concentrated distribution around fair/good with virtually no bad-quality representation. Each expansion stream targets different parts of the quality spectrum.

| Level | Base | Stream 1 | Stream 2 | Stream 3 | Combined | Gate (≥2%) |
|-------|-----:|--------:|--------:|--------:|---------:|----------:|
| **excellent** | 7.8% | 7.7% | 12.8% | 6.6% | **7.9%** | PASS |
| **good** | 49.3% | 18.0% | 24.2% | 27.9% | **32.1%** | PASS |
| **fair** | 36.6% | 21.6% | 25.3% | 49.0% | **39.8%** | PASS |
| **poor** | 5.8% | 26.5% | 22.0% | 13.9% | **14.1%** | PASS |
| **bad** | 0.6% | 26.2% | 15.7% | 2.6% | **6.1%** | PASS |

**Key improvements**:
- **bad**: 0.6% → 6.1% (10x increase) — primarily from Stream 1 heavy/historical degradation
- **poor**: 5.8% → 14.1% (2.4x increase) — from Stream 1 + Stream 2 moderate degradation
- **good**: 49.3% → 32.1% (reduced concentration) — diluted by broader expansion data

### 2.2 MOS Statistics by Stream

| Stream | N | Mean | Std | Min | Max | P25 | P50 | P75 |
|--------|---:|-----:|----:|----:|----:|----:|----:|----:|
| Base | 3,500 | 2.89 | 0.54 | 0.63 | 4.13 | 2.59 | 2.92 | 3.25 |
| Stream 1 | 1,260 | 2.54 | 1.03 | 1.41 | 4.37 | 1.49 | 2.32 | 3.71 |
| Stream 2 | 1,800 | 2.96 | 1.01 | 1.41 | 4.56 | 2.00 | 3.02 | 3.93 |
| Stream 3 | 6,603 | 3.22 | 0.48 | 1.12 | 4.88 | 3.00 | 3.00 | 3.47 |
| **Combined** | **13,163** | **3.03** | **0.69** | **0.63** | **4.88** | **2.65** | **3.00** | **3.47** |

Stream 1 provides the widest quality spread (std=1.03), which is by design — it applies 4 degradation tiers (light, moderate, heavy, historical) to high-quality base images. Stream 3 clusters around fair/good (mean=3.22) because most real documents in the source datasets have moderate quality.

---

## 3. Diversity Analysis

### 3.1 Capture Method Diversity

The baseline is 91% synthetic. The expansion introduces 4 real-world capture methods:

| Capture Method | Baseline | Tier 1 | Sources |
|----------------|----------|--------|---------|
| **Synthetic** | 91% | 40.3% | DIQA base, Stream 1, Stream 2 |
| **Camera (controlled)** | 0% | 29.1% | SmartDoc-QA (robotic arm, 2 phones) |
| **Scanner (modern)** | 0% | 8.0% | FUNSD, FUNSD+ |
| **Scanner (archival)** | 0% | 4.8% | Tobacco800 |
| **Camera (natural)** | 9% | 5.5% | RealDAE, SROIE |
| **Mixed** | 0% | 2.7% | OCR-Quality |

### 3.2 Script Diversity

The baseline is 75% Hans + 23% Latn. Stream 2 adds 24 ISO 15924 scripts with balanced distribution (~73-79 samples each):

Arab, Beng, Cyrl, Deva, Ethi, Grek, Gujr, Guru, Hans, Hant, Hebr, Jpan, Khmr, Knda, Laoo, Latn, Mlym, Mymr, Orya, Sinh, Taml, Telu, Thai, Tibt

This is critical for generalization — a DIQA model that only sees Hans/Latn will systematically misjudge quality for documents in other scripts.

### 3.3 Document Domain Diversity

| Domain | Baseline | Tier 1 Sources |
|--------|----------|----------------|
| Educational/Scientific | High | DIQA base, OCR-Quality |
| Administrative forms | None | FUNSD, FUNSD+ |
| Archival/Historical | None | Tobacco800 |
| Retail receipts | None | SROIE |
| General documents | Low | SmartDoc-QA (modern + admin + receipts) |
| Camera-captured | Low | SmartDoc-QA, RealDAE |

### 3.4 Degradation Type Coverage

| Degradation | Baseline | Stream 1 | Stream 2 | Stream 3 (natural) |
|-------------|----------|----------|----------|---------------------|
| Blur (Gaussian/motion) | Limited | Yes (4 tiers) | Yes (5 tiers) | Yes (SmartDoc-QA) |
| Noise | Limited | Yes | Yes | Yes (archival) |
| JPEG compression | No | Yes | Yes | No |
| Perspective distortion | No | No | No | Yes (SmartDoc-QA) |
| Paper aging | No | Yes (historical) | No | Yes (Tobacco800) |
| Ink degradation | No | Yes | Yes | Yes (Tobacco800) |
| Lighting variation | No | No | No | Yes (SmartDoc-QA, 5 conditions) |
| Defocus blur | No | No | No | Yes (SmartDoc-QA) |
| Bleed-through | No | Yes | Yes | Yes (RealDAE) |

### 3.5 Ground Truth Text Availability

42.0% of training samples (5,523/13,163) have ground truth text, enabling future OCR-IQA correlation analysis:

| Source | GT Text | Count | Provenance |
|--------|---------|------:|------------|
| SmartDoc-QA | Human-keyed | 3,834 | Tier 1 (manually transcribed) |
| FUNSD+ | Human-annotated | 924 | Tier 1 (form entity annotations) |
| OCR-Quality | Human-labeled | 360 | Tier 1 |
| SROIE | Human-annotated | 270 | Tier 1 (receipt entities) |
| FUNSD | Human-annotated | 135 | Tier 1 (form entity annotations) |

---

## 4. Pseudo-Label Accuracy Analysis

### 4.1 VLM Consensus Protocol

Stream 3 uses a two-model consensus with tiebreaker:

1. **Primary**: Gemini 2.0 Flash Lite (via OpenRouter)
2. **Secondary**: Qwen3-VL-235B-A22B-Instruct (via OpenRouter)
3. **Tiebreaker**: GPT-4.1 (invoked when |Δ| > 1.0 MOS)

Both models receive the same image and prompt: *"Rate the overall quality of this document image. Consider readability, clarity, and general visual quality. Choose exactly one: excellent, good, fair, poor, or bad. Respond with only one word."*

Consensus MOS is computed as:
- **Agreement** (|Δ| ≤ 1.0 MOS): mean of primary + secondary scores
- **Disagreement** (|Δ| > 1.0): median of primary + secondary + tiebreaker
- **All fail to parse**: image excluded (0 occurrences)

### 4.2 Inter-Model Agreement

| Metric | Value |
|--------|-------|
| **Total images labeled** | 7,498 |
| **Agreement rate** (|Δ| ≤ 1.0) | 99.2% (7,438/7,498) |
| **Exact agreement** (|Δ| = 0) | 49.3% (3,691/7,498) |
| **Tiebreaker invoked** | 0.7% (53/7,498) |
| **Exclusions** | 0 |
| **Parse failures** | 0 |
| **Mean |Δ|** | 0.514 MOS |
| **Max |Δ|** | 2.0 MOS |

99.2% agreement rate is strong evidence that both models are measuring the same quality construct. The 0.7% disagreement rate (53 images requiring tiebreaker) is concentrated in Tobacco800 (12 images, 1.7%) — expected, since archival scanned documents have ambiguous quality characteristics.

### 4.3 Agreement by Source Dataset

| Source | Agreement | Rate | Notes |
|--------|-----------|-----:|-------|
| FUNSD+ | 1,138/1,139 | 100% | Scanner-captured forms — unambiguous quality |
| RealDAE | 500/500 | 100% | Camera captures — clear quality signal |
| SROIE | 300/300 | 100% | Receipt images — consistent assessment |
| SmartDoc-QA | 4,218/4,260 | 99% | Controlled distortions — well-defined quality |
| FUNSD | 197/199 | 99% | Scanner-captured forms |
| OCR-Quality | 397/400 | 99% | Multilingual — slight ambiguity |
| Tobacco800 | 688/700 | 98% | Archival — most ambiguous quality |

### 4.4 Model Label Distribution

The two models exhibit different calibration tendencies:

| Level | Gemini Flash Lite | Qwen3-VL-235B | Implication |
|-------|------------------:|-------------:|-------------|
| **excellent** | 0.5% | 1.5% | Both conservative at top |
| **good** | 48.2% | 11.6% | Gemini skews higher |
| **fair** | 48.8% | 74.0% | Qwen skews to fair |
| **poor** | 2.5% | 11.3% | Qwen more willing to rate low |
| **bad** | 0.0% | 1.7% | Only Qwen uses bad |

**Calibration insight**: Gemini clusters around good/fair (97.0% combined), while Qwen3 has a broader distribution centered on fair (74.0%) with meaningful poor/bad representation. The consensus averaging mitigates each model's individual bias — Gemini prevents the labels from being too pessimistic, while Qwen prevents them from being too optimistic.

This complementary calibration is why the ensemble outperforms either model individually (wSRCC=0.777 ensemble vs 0.743 Gemini alone, 0.728 Qwen alone, from the VLM calibration study).

### 4.5 SmartDoc-QA Cross-Validation: VLM vs OCR Accuracy

SmartDoc-QA provides a unique validation opportunity: every image has both a VLM consensus label AND pre-computed OCR accuracy from two engines. This enables objective assessment of VLM label quality without circular evaluation.

**OCR accuracy distribution** (FineReader, 4,260 images):
- Mean: 29.1%, P50: 7.2%, range: 0-100%
- 62.8% of images have < 20% accuracy (heavy distortion conditions)

**Expected correlation**: Higher VLM quality scores should correlate with higher OCR accuracy. The full three-way analysis (VLM vs OCR accuracy vs DeQA prediction) will be performed post-training.

### 4.6 Label Confidence Weighting

To account for the inherent noise difference between label sources, training samples are weighted:

| Label Source | Weight | Rationale |
|-------------|--------|-----------|
| **Human MOS** (base) | 1.0 | Direct human quality judgments — gold standard |
| **Deterministic** (S1, S2) | 0.7 | Computed from known degradation parameters — reliable but synthetic |
| **VLM consensus** (S3) | 0.5 | Model-predicted — highest noise, but consensus averaging reduces variance |

The SoftKL loss function uses these weights to scale each sample's contribution to the training loss, ensuring that noisier labels have less influence on model parameters.

### 4.7 Soft-Label Generation

All labels are converted to 5-bin soft-label distributions using Gaussian CDF integration:

```text
MOS → N(MOS, σ²) → CDF integration over bins [4.5, 3.5, 2.5, 1.5, 0.5] → [p_excellent, p_good, p_fair, p_poor, p_bad]
```

- **Base**: σ from human annotation variance (per-image std)
- **Deterministic** (S1, S2): σ = 0.8 (matching DeQA's σ_pseudo)
- **VLM consensus**: σ = 0.4 + |Δ|/2 (minimum 0.4, increases with disagreement)

Higher σ spreads probability mass across adjacent levels, encoding uncertainty. VLM labels with low inter-model disagreement get tighter distributions (σ ≈ 0.4), while high-disagreement labels get broader distributions (σ up to 1.25), naturally downweighting uncertain samples.

---

## 5. Validation Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| excellent mass | ≥ 2% | 7.9% | PASS |
| good mass | ≥ 2% | 32.1% | PASS |
| fair mass | ≥ 2% | 39.8% | PASS |
| poor mass | ≥ 2% | 14.1% | PASS |
| bad mass | ≥ 2% | 6.1% | PASS |
| ID wSRCC delta | ≥ -0.01 | Pending | Post-training |
| OOD wSRCC delta | ≥ 0.02 | Pending | Post-training |
| Tail SRCC | ≥ 0.50 | Pending | Post-training |

---

## 6. Risk Assessment

### 6.1 Identified Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SmartDoc-QA content redundancy (30 docs × 142 captures) | Medium | Document-level splits prevent leakage; 21 train docs provide sufficient content diversity |
| VLM calibration bias (Gemini=good/fair, Qwen=fair) | Low | Consensus averaging corrects individual model biases; weight=0.5 limits influence |
| Stream 3 dominance (50.2% of training) | Medium | Lower weight (0.5) reduces effective contribution; base data anchors quality scale |
| Latin-script bias in VLM sources | Low | Stream 2 provides 24-script coverage; OCR-Quality adds Hans diversity |
| Controlled vs natural degradation gap | Low | SmartDoc-QA bridges gap (controlled captures of real documents); Tobacco800/RealDAE provide natural degradation |

### 6.2 Post-Training Validation Plan

1. **ID evaluation**: DIQA-5000 val/test splits (550 each) — wSRCC should not degrade by > 0.01
2. **OOD evaluation**: SmartDoc-QA test split (852 images, 6 docs) — wSRCC should improve by > 0.02
3. **Tail evaluation**: SRCC on images with MOS < 2.0 or MOS > 4.0 — target ≥ 0.50
4. **OCR-IQA correlation**: SmartDoc-QA test split — DeQA MOS vs FineReader char accuracy SRCC
5. **Cross-model consistency**: Compare mPLUG-Owl2 predictions with VLM consensus labels on holdout

---

## 7. File Manifest

| File | Location | Records | Description |
|------|----------|--------:|-------------|
| `train_overall.json` | `DIQA-5000_1/` | 13,163 | Combined training manifest |
| `holdout_overall.json` | `DIQA-5000_1/` | 1,072 | 10% holdout (stratified by source) |
| `stream1_records.json` | `DIQA-5000_1/` | 1,400 | Stream 1 records (pre-holdout) |
| `stream2_records.json` | `DIQA-5000_1/` | 2,000 | Stream 2 records (pre-holdout) |
| `stream3_records.json` | `DIQA-5000_1/` | 7,335 | Stream 3 train-split records (pre-holdout) |
| `stream3_all_vlm_records.json` | `DIQA-5000_1/` | 7,498 | All VLM labels (all splits) |
| `vlm_consensus_results.json` | `DIQA-5000_1/` | 7,498 | Raw VLM checkpoint data |
| `manifest.json` | `DIQA-5000_1/` | — | Dataset provenance and metadata |
| `smartdoc_qa_splits.json` | `smartdoc-qa/splits/` | — | Document-level split manifest |

**Image directories** (under `DIQA-5000_1/images/`):

| Directory | Files | Source |
|-----------|------:|--------|
| `stream1_degradation/` | 1,400 | Degraded DIQA base images |
| `stream2_synth_multiscript/` | 2,000 | Fresh synthetic documents |
| `stream3_smartdoc_qa/` | 4,259 | SmartDoc-QA captures |
| `stream3_funsd_plus/` | 1,139 | FUNSD+ form images |
| `stream3_tobacco800/` | 700 | Archival scanned documents |
| `stream3_realdae/` | 500 | Camera-captured documents |
| `stream3_ocr_quality/` | 400 | Multilingual documents |
| `stream3_sroie/` | 300 | Receipt images |
| `stream3_funsd/` | 199 | FUNSD form images |

---

## 8. Training Configuration

Training script: `scripts/train_tier1_lora.sh`

```bash
LOAD=/path/to/mplug-owl2-base OUTPUT=./checkpoints/tier1 sh scripts/train_tier1_lora.sh
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| dataset_type | single | No pair ranking for Tier 1 (diverse sources) |
| weight_softkl | 1.0 | Primary loss — soft-label distribution matching |
| weight_next_token | 0.005 | Low autoregressive weight — focus on quality prediction |
| weight_rank | 0.0 | Disabled — pair ranking requires same-source pairs |
| learning_rate | 2e-5 | Standard LoRA fine-tuning rate |
| epochs | 3 | Sufficient for LoRA convergence |
| save_strategy | steps (500) | Checkpoint every 500 steps for recovery |
