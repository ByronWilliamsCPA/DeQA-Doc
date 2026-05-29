# 04 Architecture and Structure

The core training/inference package (`DeQA-Score/src/`) has clean internal layering: `src/constants.py` is the single source of truth for quality-level constants and is imported by the 10 modules that make up the model, train, datasets, and evaluate layers. The structural debt sits at the seams between that core and the satellite tiers (`research/`, `results/`, `modal/`, `expansion/`), which re-declare the same constants locally instead of importing them and reach into the core via `sys.path` insertion rather than a package install. The most serious instance is a semantic split in the dimension name (`color` vs `color_fidelity`) across script tiers, which can silently mismatch dict keys. The prior `PROJECT_REVIEW_REPORT.md` (2026-03-09) named this same "no single source of truth for constants" as its top finding; it persists 2.5 months later.

## ARCH-01 Quality-level constants re-declared in 9+ locations instead of imported
- Severity: High
- Effort: M
- CVE:
- Affected files: `DeQA-Score/src/expansion/iqa_to_deqa.py:40`, `src/uncertainty/cross_validator.py:26`, `src/uncertainty/format_training_data.py:41`, `src/uncertainty/metadata_schema.py:29`, `modal/run_deqa_ocr_iqa.py:87`, `modal/benchmark_synthetic_ood.py:672`, `research/ocr_iqa_correlation/deqa/scorer_wrapper.py:21`, plus the canonical `src/constants.py:16`
- Evidence: `LEVEL_NAMES = ["excellent", "good", "fair", "poor", "bad"]` is redefined verbatim in at least 7 modules outside `src/constants.py`. `src/constants.py` already exports it and is correctly imported by the core (`src/train/loss.py`, `src/datasets/*`, `src/evaluate/*`). The satellite tiers do not import it; they copy it. Any change to the level vocabulary must be made in 8 places or the copies drift.
- Recommendation: Make the satellite modules import `LEVEL_NAMES`/`LEVEL_PREFIX` from `src.constants` (they already use `sys.path` to reach `src`). Delete the local copies.

## ARCH-02 Dimension name split: "color" vs "color_fidelity"
- Severity: High
- Effort: S
- CVE:
- Affected files: `research/papers/shared/constants.py:18` (`["overall", "sharpness", "color_fidelity"]`), versus `DeQA-Score/src/uncertainty/pseudo_label.py:27`, `modal/siglip2_v2_model.py:30`, `research/threshold_sensitivity/run_sweep.py:48`, `results/siglip2_v2_pseudo_label_validation/*.py` (all `("overall", "sharpness", "color")`)
- Evidence: `git grep 'DIMENSIONS ='` shows two incompatible value sets for the third dimension, `color` in the pipeline and `color_fidelity` in the papers tier, alongside dict-shaped `DIMENSIONS` in `research/vlm_calibration/*` and `results/siglip2_diqa5000/calibrate_isotonic.py`. A dict keyed on one spelling will silently miss data keyed on the other.
- Recommendation: Pick one canonical dimension key (the pipeline uses `color`), add it to `src/constants.py` as `DIMENSIONS`, and import everywhere. Treat the `color_fidelity` spelling as a bug to reconcile.

## ARCH-03 Satellite tiers reach into the core via sys.path insertion
- Severity: Medium
- Effort: M
- CVE:
- Affected files: 35 `sys.path.insert`/`append` sites across `research/**`, `results/**`, `modal/**`, and `DeQA-Score/src/**`
- Evidence: `git grep -cE 'sys.path.(insert|append)'` returns 35 in new code. `research/threshold_sensitivity/run_sweep.py:39-43` inserts a path then imports `from src.uncertainty...` with `# noqa: E402`. The core is not installed as a package into these environments; scripts mutate `sys.path` at runtime to find it.
- Recommendation: Install `DeQA-Score` as an editable package (`uv pip install -e DeQA-Score`) in the research/modal environments so imports resolve without path hacks. Removes the `E402` suppressions too.

## ARCH-04 research/ and results/ overlap in purpose
- Severity: Low
- Effort: M
- CVE:
- Affected files: `research/` (177 files), `results/` (96 files)
- Evidence: Both hold experiment runners plus outputs (`results/vlm_teacher_eval/run_eval.py` and `research/vlm_calibration/evaluate_calibration.py` are both eval drivers; both trees mix `.py` runners with `.json`/`.csv`/`.png` outputs). The boundary between "research" and "results" is not stated in `CLAUDE.md` and is not observable from contents.
- Recommendation: Document the intended split (for example research = exploration, results = reproducible artifacts for the paper) in `CLAUDE.md`, or merge them. Low priority while the work is active.

## ARCH-05 Two-backend split relies on copy-into-install patching for Qwen
- Severity: Medium
- Effort: M
- CVE:
- Affected files: `Llamafactory/src/llamafactory/**` (16 files)
- Evidence: `CLAUDE.md` states the `Llamafactory/` files are "patches to be copied into a LLaMA-Factory installation (paths match)". There is no pinned LLaMA-Factory version or automated apply step; the patch set (`train/sft/loss.py`, `data/collator.py`, etc.) silently breaks if upstream LLaMA-Factory refactors the touched files.
- Recommendation: Pin the exact LLaMA-Factory commit the patches target and record it in `Llamafactory/README` or `CLAUDE.md`, plus a one-line apply script. Without the pin, the Qwen backend is not reproducible.

## Clean areas (one line each)
- Core package layering: `src/constants.py` is a real single source of truth, imported by 10 core modules (`git grep 'from src.constants'`); no circular imports observed in the core.
- The 24-module `src/uncertainty/` package is cohesive: each module maps to one pipeline stage (pseudo-label, fusion, cross-validation, metadata IO/schema, OOD), not a grab-bag.
