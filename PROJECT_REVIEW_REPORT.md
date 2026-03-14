# DeQA-Doc Project Review Report

**Date**: 2026-03-09
**Scope**: Systematic review of code correctness, architecture, testing, and cross-component consistency
**Codebase**: ~15K lines of new/modified code across training, uncertainty pipeline, evaluation, and research scripts

---

## Executive Summary

The review identified **8 critical**, **10 high**, **16 medium**, and **14 low** severity findings across 7 review areas. The codebase has a mathematically sound uncertainty pipeline but suffers from **integration fragility** — critical constants (level names, prefixes, token IDs) are defined independently in 5-7 locations with no single source of truth. Three findings may affect currently trained models:

### Top 5 Critical Findings

| # | Finding | Impact |
|---|---------|--------|
| 1 | `level_prefix` mismatch between training scripts and inference code | Trained models may extract quality scores at wrong token position |
| 2 | `find_prefix()` silently returns position 0 when prefix not found | Garbage gradients corrupt training without error signal |
| 3 | `torch.log(preds)` without epsilon clamping under fp16/bf16 | NaN loss / training crash |
| 4 | Infinite retry loops in dataset `__getitem__` | Training hangs indefinitely on systematic errors |
| 5 | `infer.sh` references nonexistent `iqa_eval_prompt.py` | Inference script broken |

### Overall Risk Assessment

| Area | Risk | Key Issue |
|------|------|-----------|
| Training Loss (mPLUG-Owl2) | **HIGH** | Silent `find_prefix` failure, NaN risk |
| Training Loss (Qwen2.5-VL) | **HIGH** | Missing InLevel loss, print in hot path, silent failures |
| Data Pipeline | **HIGH** | Infinite loops, sentinel values flowing to loss |
| Model Loading & Inference | **CRITICAL** | Hardcoded cloud path, prompt mismatch |
| Uncertainty Pipeline (algorithms) | **LOW** | Mathematically sound, minor cleanup items |
| Uncertainty Pipeline (data/tests) | **MEDIUM** | Zero test coverage for VLM validator and pseudo-label orchestrator |
| Research & Analysis Scripts | **LOW-MEDIUM** | Good practices, minor parsing and performance issues |
| Integration & Architecture | **CRITICAL** | Constants duplication, prefix divergence, no integration tests |

---

## Findings by Severity

### CRITICAL (8 findings)

**C1. `level_prefix` mismatch between training shell scripts and all other code**
- **Files**: `scripts/train.sh:11` (`"quality of the image is"`) vs `iqa_eval.py:68`, `scorer.py:18`, `Llamafactory workflow.py:61` (`"The quality of the image is"`)
- **Impact**: If models were trained with the shell script prefix (missing "The"), inference extracts logits at the wrong position, producing silently wrong quality scores.
- **Fix**: Audit which prefix was used in successful training runs. Standardize to a single constant in `src/constants.py`.

**C2. `find_prefix()` silent failure returns position 0**
- **Files**: `Llamafactory/train/sft/loss.py:274` (returns 0), `DeQA-Score/src/train/loss.py:53-65` (returns empty tensor)
- **Impact**: When prefix is absent from a label sequence (truncation, corruption), SoftKL loss operates on BOS/padding token logits — producing garbage gradients that corrupt training with no error signal.
- **Fix**: Raise an exception or skip the sample from SoftKL computation. The original `utils.py:31` correctly asserts.

**C3. `torch.log(preds)` without epsilon clamping**
- **Files**: `modeling_mplug_owl2.py:403`, `loss.py:85`, `Llamafactory loss.py:229`
- **Impact**: Under fp16/bf16 mixed-precision training, softmax outputs can underflow to 0.0, making `log(0) = -inf` → NaN loss propagation.
- **Fix**: Replace `torch.log(torch.softmax(logits, dim=1))` with `F.log_softmax(logits, dim=1)` (numerically stable).

**C4. Infinite retry loop in dataset `__getitem__`**
- **Files**: `single_dataset.py:70`, `pair_dataset.py:80`
- **Impact**: `while True:` with exception retry and no counter. Wrong `image_folder` path → infinite busy-loop consuming 100% CPU. Training hangs forever with no diagnostics.
- **Fix**: Add max-retry counter (e.g., 50). Raise `RuntimeError` after exhausting retries.

**C5. `infer.sh` references nonexistent `iqa_eval_prompt.py`**
- **File**: `scripts/infer.sh:3`
- **Impact**: Inference script fails immediately with `FileNotFoundError`.
- **Fix**: Change to `src/evaluate/iqa_eval.py`.

**C6. `make_data_module()` signature mismatch**
- **File**: `datasets/__init__.py:5` requires 3 args, `train_mem.py:478` passes 2
- **Impact**: `TypeError` at runtime. Either this code path is dead or was broken by a recent edit.
- **Fix**: Remove `training_args` from the dispatcher signature or add `training_args=None` default.

**C7. `vlm_validator.py` `_parse_vlm_response` false-positive matching**
- **File**: `vlm_validator.py:116`
- **Impact**: `text.startswith(level)` matches "badly" as "bad", "goodness" as "good". Since VLM veto decisions permanently reject training samples, false-positive parsing corrupts training data quality.
- **Fix**: Require word boundary: `text == level or text.startswith(level + " ")`.

**C8. Zero test coverage for `vlm_validator.py`**
- **File**: No `test_vlm_validator.py` exists
- **Impact**: 315 lines of code making permanent veto decisions with external API calls, text parsing edge cases, and budget accounting — completely untested.
- **Fix**: Create test file with mocked httpx calls covering parse edge cases, API errors, budget tracking.

---

### HIGH (10 findings)

**H1. LlamaFactory missing InLevel loss and multi-attribute support**
- `Llamafactory/train/sft/loss.py` omits InLevel loss (probability concentration on quality tokens) present in mPLUG-Owl2. LlamaFactory is single-attribute only.
- Models trained on different backends learn different loss landscapes.
- **Fix**: Port InLevel loss. Document intentional backend differences.

**H2. `level_probs` sentinel `[-10000]*5` flows into SoftKL loss**
- `single_dataset.py:175` sets sentinel; `modeling_mplug_owl2.py:400` uses it without validation.
- KL divergence with -10000 target values produces mathematically meaningless gradients.
- **Fix**: Guard in loss function: skip SoftKL if any `level_probs` value equals -10000.

**H3. Print statement in hot training path**
- `Llamafactory/train/sft/loss.py:157`: `print("成功计算loss_kl:",loss_kl)` every batch.
- Floods stdout, degrades training performance.
- **Fix**: Remove entirely.

**H4. `find_prefix` has 4 implementations with behavioral divergence**
- `utils.py` (asserts), `loss.py` (returns end position), `Llamafactory loss.py` (returns 0 on miss), `test_find_prefix.py` (standalone copy)
- Different return semantics (prefix start vs end) and different failure modes.
- **Fix**: Consolidate to single canonical implementation in `utils.py`.

**H5. `expand2square` duplicated in 7+ locations**
- Includes one defined inside a per-image loop (`iqa_eval.py:110`).
- `conversation.py` variant hardcodes different background color.
- **Fix**: Import from `src.mm_utils` everywhere.

**H6. Token ID extraction assumes `[BOS, token]` pattern**
- `scorer.py:20`, `iqa_eval.py:72`, `train_mem.py:429` all use `id_[1]`.
- Only correct for LLaMA-2 tokenizer. Silently wrong for other tokenizers.
- **Fix**: Create `get_level_token_ids()` utility handling multiple tokenizer families.

**H7. Hardcoded `/ossfs/workspace/...` path in builder.py:62**
- Breaks `Scorer` API and default `iqa_eval.py` on all non-Alibaba-Cloud environments.
- **Fix**: Fall back to `model_path` when `preprocessor_path` is None.

**H8. CUDA-gated `src/__init__.py` blocks importing uncertainty pipeline**
- 4000 lines of pure Python/NumPy code unusable without GPU.
- **Fix**: Lazy-load model in `__init__.py` via `__getattr__`.

**H9. `cal_distribution_gap.py:123` passes `logits=` when `use_openset_probs=True`**
- `cal_score()` asserts `logits is None` in this mode, expects `probs=`.
- Runtime `AssertionError` when using `--use_openset_probs`.
- **Fix**: Change to `probs=pred_meta["probs"]`.

**H10. No test coverage for `pseudo_label.py` orchestration**
- 320-line top-level pipeline orchestrator — completely untested.
- Individual components tested, but integration wiring is not.
- **Fix**: Create `test_pseudo_label.py` with mocked dependencies.

---

### MEDIUM (16 findings)

| ID | Finding | File(s) |
|----|---------|---------|
| M1 | Prompt template divergence between `scorer.py` and `iqa_eval.py` | `scorer.py:18`, `iqa_eval.py:60-68` |
| M2 | Level names defined independently in 7+ locations with no shared constant | Multiple (see Chunk 7) |
| M3 | Token ID extraction inconsistency — training asserts len=2, inference does not | `train_mem.py:427`, `scorer.py:20` |
| M4 | Silent exception swallowing with `print(ex)` in data loading | `single_dataset.py:98-190` |
| M5 | PairDataset no validation for different-score pairs (ranking loss needs this) | `pair_dataset.py:96-102` |
| M6 | `get_subitem` mutates caller's `kwargs` dict in-place | `modeling_mplug_owl2.py:561-586` |
| M7 | LlamaFactory `AlpacaDatasetConverter` does not preserve `level_probs` | `converter.py:70-116` |
| M8 | LlamaFactory premature tensor conversion in Arrow dataset | `loader.py:112-115` |
| M9 | `metadata_convert.py:to_training_record` loses `gt_score_norm` and `std_norm` | `metadata_convert.py:423` |
| M10 | Round-trip test does not verify `gt_score_norm` preservation | `test_metadata_schema.py:802-833` |
| M11 | `merge_records` does not preserve `spread` field | `metadata_io.py:136-143` |
| M12 | `format_training_data.py` and `metadata_convert.py` duplicate training record generation | Both files |
| M13 | `icecream` debug imports in 5 production modules | `mm_utils.py`, `trainer.py`, `modeling_mplug_owl2.py`, `visual_encoder.py` |
| M14 | `torch.load` without `weights_only=True` | `builder.py:92,105` |
| M15 | VLMValidator has no retry logic for rate limits | `vlm_validator.py:159-226` |
| M16 | Integration tests are shallow — check existence, not behavioral contracts | `test_contract.py`, `test_sonnet_smoke.py` |

---

### LOW (14 findings)

| ID | Finding | File(s) |
|----|---------|---------|
| L1 | `data_dict * data_weight` duplicates references not copies | `single_dataset.py:32` |
| L2 | Unused `DeQAScoreLoss` class creates confusion | `loss.py:11-162` |
| L3 | Mutable default argument in `DeQAScoreLoss.__init__` | `loss.py:20-24` |
| L4 | Bare `except:` clauses | `iqa_eval.py:100`, `utils.py:22-27` |
| L5 | File handles not closed with context manager | `single_dataset.py:31` |
| L6 | `LOAD_TRUNCATED_IMAGES = True` masks data quality issues | `utils.py:7` |
| L7 | Typo `tokem_dim` in builder.py | `builder.py:77` |
| L8 | SRCC/PLCC code commented out in cal_plcc_srcc.py | `cal_plcc_srcc.py:120-124` |
| L9 | `--with-prob` uses `type=bool` (always True for any string) | `iqa_eval.py:173` |
| L10 | Dead expression `jsds_sorted[end-1]` in fusion calibration | `fusion.py:401` |
| L11 | Docstring contradicts implementation in `transform_batch` | `model_normalizer.py:114-131` |
| L12 | `auto_upgrade()` uses `input()` — blocks headless environments | `utils.py:45` |
| L13 | `.env` loading at module import time in test file | `test_sonnet_smoke.py:62` |
| L14 | `np.std` uses population std for small sample sizes | `run_eval.py:431` |

---

## Findings by Area

### Training Pipeline (mPLUG-Owl2)
- C2 (find_prefix silent failure), C3 (log NaN), C4 (infinite loop), C6 (signature mismatch)
- H2 (sentinel in loss), H4 (find_prefix divergence)
- Core loss computation (SoftKL, CE offset) is mathematically correct

### Training Pipeline (Qwen2.5-VL / LlamaFactory)
- C2 (find_prefix returns 0), C3 (log NaN)
- H1 (missing InLevel loss), H3 (print in hot path)
- Single-attribute only, undocumented divergence from mPLUG-Owl2

### Data Pipeline
- C4 (infinite loops), C6 (signature mismatch)
- H2 (sentinel values), M4 (silent exceptions), M5 (degenerate pairs)

### Model Loading & Inference
- C1 (prefix mismatch), C5 (broken infer.sh)
- H7 (hardcoded path), H9 (logits/probs mixup)
- M1 (prompt divergence)

### Uncertainty Pipeline
- C7 (parse false-positive), C8 (zero VLM test coverage)
- H10 (zero pseudo_label test coverage)
- Core algorithms verified correct — level ordering, CDF binning, JSD, fusion logic all sound

### Research & Analysis Scripts
- No critical issues. Good API key handling. Minor parsing and performance items.

### Architecture / Integration
- C1 (prefix mismatch), H4 (find_prefix x4), H5 (expand2square x7), H6 (token ID fragility)
- H8 (CUDA-gated imports), M2 (level names x7), M16 (shallow integration tests)

---

## Recommendations (Priority Order)

### P0 — Verify Immediately
1. **Verify `level_prefix` used in successful training runs** — Check training logs or saved config to determine if models were trained with `"quality of the image is"` or `"The quality of the image is"`. If the shell script prefix was used, inference code needs to match.

### P1 — Fix Before Next Training Run
2. **Replace `torch.log(softmax(...))` with `F.log_softmax()`** in all 3 locations
3. **Add max-retry to dataset `__getitem__` loops** (50 retries, then raise)
4. **Fix `find_prefix` silent failure** — raise or skip sample, never return position 0
5. **Fix `make_data_module` signature** — align dispatcher with callers
6. **Remove `print("成功计算loss_kl:")` from Llamafactory hot path**
7. **Fix `infer.sh` to reference correct file**

### P2 — Fix Before Next Release
8. **Centralize constants** — add `LEVEL_NAMES`, `LEVEL_SCORES`, `LEVEL_PREFIX` to `src/constants.py`
9. **Consolidate `find_prefix`** — single implementation in `utils.py`, import everywhere
10. **Consolidate `expand2square`** — import from `src.mm_utils`
11. **Fix builder.py hardcoded path** — fall back to `model_path`
12. **Fix `vlm_validator` `startswith` parsing** — require word boundary
13. **Add guard for `level_probs` sentinel** in loss function
14. **Fix `cal_distribution_gap.py:123`** — `logits=` → `probs=`

### P3 — Improve Quality
15. **Create `test_vlm_validator.py`** and **`test_pseudo_label.py`**
16. **Lazy-load model in `src/__init__.py`** to unblock uncertainty pipeline imports
17. **Create canonical `get_level_token_ids()` utility**
18. **Add integration contract tests** for level name/prefix/token consistency
19. **Remove `icecream` imports** from production modules
20. **Add `weights_only=True` to `torch.load` calls**

---

## Verification Checklist

After implementing fixes:
- [ ] Run existing tests: `cd DeQA-Score && .venv/bin/python -m pytest tests/uncertainty/ -v --tb=short` (87 tests, all should pass)
- [ ] Verify level_prefix in a training config matches inference code
- [ ] Run `python -c "from src.uncertainty.metadata_schema import ImageMetadataRecord"` without GPU (after lazy-load fix)
- [ ] Confirm `find_prefix` raises on missing prefix in both backends
- [ ] Run `scripts/infer.sh` after fixing file reference
- [ ] Verify `F.log_softmax` produces same results as `log(softmax(...))` for normal inputs
