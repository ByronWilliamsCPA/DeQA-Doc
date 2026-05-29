# 02 Legacy Code Patterns

The repo is clean on imminent runtime-breaking deprecations: zero hits for `datetime.utcnow()`, `pkg_resources`, `import imp`, `distutils`, `asyncio.get_event_loop`, or `np.float`/`np.int`/`np.bool` aliases across all 197 `.py` files. The actual legacy debt sits in the vendored mPLUG-Owl2 model files (`DeQA-Score/src/model/`), which copy transformers private internals and are hard-pinned to `transformers==4.36.1` (LEG-01, the only High). Everything else is style-level: upstream code uses `typing.List/Dict/Optional` generics and `os.path`, new project code mostly does not. New code (`uncertainty/`, `research/`, `results/`, `modal/`) is in good shape: near-zero commented-out code, no real `%`-formatting, pathlib-first, and its pydantic v1 usage is deliberate and documented against the `pydantic<2` pin. Findings are weighted so vendored-upstream idioms stay Low unless they couple to the runtime pin.

## LEG-01 Vendored transformers internals coupled to the transformers==4.36.1 pin
- Severity: High
- Effort: L (re-vendor or upstream-track on each transformers bump; large surface)
- CVE:
- Affected files: `DeQA-Score/src/model/modeling_llama2.py` (869 LOC), `DeQA-Score/src/model/modeling_attn_mask_utils.py` (11.4 KB), `DeQA-Score/src/model/visual_encoder.py` (1017 LOC), `DeQA-Score/src/model/modeling_mplug_owl2.py` (772 LOC), `DeQA-Score/src/model/configuration_mplug_owl2.py`
- Evidence: `modeling_llama2.py:14-23` imports private symbols from `transformers.models.llama.modeling_llama` (`LlamaRotaryEmbedding`, `apply_rotary_pos_emb`, `repeat_kv`, `LlamaMLP`, `LlamaRMSNorm`, `is_flash_attn_greater_or_equal_2_10`). `modeling_llama2.py:35` does `from transformers.models.llama.modeling_llama import *`. `modeling_attn_mask_utils.py` is a verbatim copy of HuggingFace's attn-mask helpers (`# Copyright 2023 The HuggingFace Team`) providing `_make_causal_mask`/`_expand_mask`/`_prepare_4d_causal_attention_mask` (`modeling_attn_mask_utils.py:120,150,164`), symbols that were refactored/moved in later transformers. `modeling_llama2.py:859` monkeypatches `transformers.models.llama.modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2`. `pyproject.toml:18` pins `transformers==4.36.1`. Any transformers upgrade silently breaks these private imports and the monkeypatch.
- Recommendation: Keep the pin documented as load-bearing in CLAUDE.md (already partly noted), and add a comment in `modeling_llama2.py` recording the exact transformers version these internals track so the drift risk is explicit on the next bump. Do not attempt to unvendor onto a newer transformers without re-running the training pipeline.

## LEG-02 sys.path.insert import hack in vendored model loader
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `DeQA-Score/src/model/modeling_llama2.py`
- Evidence: `modeling_llama2.py:30-31` runs `dir_path = os.path.dirname(os.path.realpath(__file__)); sys.path.insert(0, dir_path)` before a `try/except` import of `.modeling_attn_mask_utils`. Mutating `sys.path` at import time can shadow same-named modules and is fragile under packaging. Vendored upstream, so Medium not High.
- Recommendation: Leave as-is unless packaging changes; the relative import (`from .modeling_attn_mask_utils import ...`) already works, making the `sys.path` insert redundant. Low-risk removal if the file is ever touched.

## LEG-03 typing.List/Dict/Optional generics instead of builtin generics / X | None
- Severity: Low
- Effort: M (mechanical but spread across files)
- CVE:
- Affected files: upstream `DeQA-Score/src/model/*` (91 hits), `Llamafactory/*` (92), `DeQA-Score/src/train/*` (12), `DeQA-Score/src/datasets/*` (8); new code `DeQA-Score/src/uncertainty/*` (59)
- Evidence: counts of `List[`/`Dict[`/`Optional[`/`Tuple[`/`Union[`: upstream model=91, Llamafactory=92, train=12, datasets=8; new uncertainty=59, research=4, modal=3. Runtime is 3.11 where `list[...]`, `dict[...]`, and `X | None` are available, so `typing` generics are superseded for new code.
- Recommendation: For new code only, prefer builtin generics and `X | None`. The 59 hits in `uncertainty/` are the actionable target; upstream (model/Llamafactory) is exempt per the vendoring policy and stays as-is.

## LEG-04 os.path usage where pathlib fits (mixed in new code)
- Severity: Low
- Effort: S
- CVE:
- Affected files: new code `modal/*` (7 `os.path` hits, 10 files already use pathlib), `DeQA-Score/src/uncertainty/*` (2 `os.path`, 9 pathlib files); upstream model=22, evaluate=10, datasets=9, Llamafactory=11
- Evidence: per-dir `os.path.` counts above. New dirs are mostly pathlib-first (`research`: 0 os.path / 38 pathlib files; `results`: 0/17; `expansion`: 0/5). Only `modal/` and `uncertainty/` mix both.
- Recommendation: Convert the residual `os.path` calls in `modal/` and `uncertainty/` to `pathlib.Path` for consistency with the rest of the new code. Upstream exempt.

## LEG-05 Bare except clauses in upstream inference/data paths
- Severity: Low
- Effort: S
- CVE:
- Affected files: `DeQA-Score/src/datasets/utils.py:26`, `DeQA-Score/src/evaluate/iqa_eval.py:100`, `DeQA-Score/src/evaluate/iqa_eval_qwen.py:126`
- Evidence: 3 occurrences of `except:` (bare) catching all exceptions including `KeyboardInterrupt`/`SystemExit`. All in vendored/upstream-style code.
- Recommendation: If these files are ever modified, narrow to `except Exception:` at minimum. Not worth a standalone change given the read-only upstream policy.

## LEG-06 .format()/%-formatting residue (minimal, new code)
- Severity: Low
- Effort: S
- CVE:
- Affected files: `research/threshold_sensitivity/run_sweep.py:760,923`, `results/vlm_teacher_eval/prompts.py:71`, `DeQA-Score/src/uncertainty/format_training_data.py:89`, `DeQA-Score/src/uncertainty/metadata_convert.py:401`
- Evidence: Real `%`-operator formatting in new code = 0 (the `%s` greps were all f-strings or format-spec false positives). `.format()` survives in 5 spots, several on reusable template strings (`answer_template.format(...)`, `SYSTEM_PROMPT_TEMPLATE.format(...)`) where `.format` on a stored template is idiomatic.
- Recommendation: The two `run_sweep.py` lines could be f-strings; the template-based `.format()` calls are fine to keep. Trivial, optional.

## LEG-07 pydantic v1 API (deliberate, documented)
- Severity: Low
- Effort: S (no action now; revisit when pin lifts)
- CVE:
- Affected files: `DeQA-Score/src/uncertainty/metadata_schema.py` (multiple `class Config:`, `validator(...)`), `DeQA-Score/src/uncertainty/metadata_io.py` (`.dict()` at lines 130,147,154,156,162,166,169)
- Evidence: `metadata_schema.py:12` docstring states "Uses Pydantic v1 API (matching the pinned `pydantic<2` constraint)"; `metadata_schema.py:24` imports `validator` (v1 name); `.dict()` (v1) used in `metadata_io.py`. `pyproject.toml:25` pins `pydantic<2,>=1` because mPLUG-Owl2 requires it (`pyproject.toml:67-70`). So this is a forced, documented choice, not accidental debt.
- Recommendation: No change. When the transformers/pydantic pin is eventually lifted, migrate `validator`->`field_validator` and `.dict()`->`.model_dump()`. Track alongside LEG-01.

## Clean areas (one line each)
- Deprecated stdlib/numpy: zero hits for `utcnow`, `pkg_resources`, `imp`, `distutils`, `get_event_loop`, `np.float`/`np.int`/`np.bool`.
- Commented-out code: new code near-zero (uncertainty 1, research 2, results 1, modal 1); concentration is upstream-only (model 26, Llamafactory 19) and expected.
- Mutable default args (`=[]`/`={}`): zero in new code.
- Resolved feature flags: none found; the one toggle (`use_deqa_loss`, `Llamafactory/.../finetuning_args.py:406`) is an active, intended training option, not dead.
- torch.load in `modal/benchmark_synthetic_ood.py:294,507` uses explicit `weights_only=False` (intentional for full-checkpoint loads), not a deprecation.
