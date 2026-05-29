# 03 Code Quality and Maintainability

The production pipeline (`DeQA-Score/src/uncertainty/`) is clean: ruff passes with no errors and only 2 functions exceed cyclomatic complexity 10. The maintainability tax is concentrated in the one-off scripts under `research/`, `results/`, and `modal/`, which carry 37 of the 40 high-complexity functions and a recurring `main()`-does-everything shape. Type-escape hatches are few (19 `type: ignore`, 6 `noqa`) and almost all are justified by inline reason codes. The test suite is sizable (339 test functions) but unmeasured in CI: the coverage workflow consumes an artifact that CI never produces, so there is no coverage figure for this repo.

Tooling note: radon and vulture were not available; complexity figures come from `ruff --select C901`. pytest could not collect under the bare interpreter (torch not installed), so test counts are from `git grep`, not a live run.

## CQ-01 No real coverage measurement despite a coverage pipeline
- Severity: High
- Effort: M
- CVE:
- Affected files: `.github/workflows/coverage.yml:30-33`, `.github/workflows/ci.yml` (quality job), `.github/workflows/python-compatibility.yml:46` (`coverage-report: false`)
- Evidence: `coverage.yml` downloads artifact `coverage-reports` / `coverage.xml` on CI success, but `ci.yml` runs no pytest and uploads no coverage artifact (`grep coverage-reports .github/workflows/ci.yml` returns nothing), and `python-compatibility.yml` sets `coverage-report: false`. The coverage upload is therefore vacuous; no coverage percentage exists for the repo.
- Recommendation: Either run the non-torch test subset in CI with `pytest --cov` and upload `coverage.xml` under the expected artifact name, or remove `coverage.yml` so it stops implying a gate that does not exist.

## CQ-02 High-complexity functions concentrated in experiment scripts
- Severity: Medium
- Effort: L (refactor across many one-off scripts; low payoff per file)
- CVE:
- Affected files: `modal/` (13 functions >10), `results/` (13), `research/` (11), `DeQA-Score/src/expansion/` (1), `DeQA-Score/src/uncertainty/` (2)
- Evidence: `ruff --select C901` (threshold 10) reports 40 functions over the limit. Worst offenders: `dry_run_check` (30), `train` (27), `main` in two scripts (27 and 24), `run_evaluations` (23), `compute_all_metrics` (22), `run_deqa_scoring` (22). Most are monolithic `main()` bodies in `research`/`results`/`modal`.
- Recommendation: Leave the one-off analysis scripts as-is (they run once and are not maintained). For the 3 in `src/` (`expansion` 1, `uncertainty` 2), extract helpers when next touched. Do not invest in refactoring the script tier.

## CQ-03 Duplicated experiment-harness scaffolding across research/results/modal
- Severity: Medium
- Effort: M
- CVE:
- Affected files: `research/**`, `results/**`, `modal/**` (multiple `run_*.py`, `evaluate_*.py`, `analyze_*.py`)
- Evidence: Repeated `main()` bodies that parse args, load env, iterate models x images, write JSON, and print a markdown table appear across `results/vlm_teacher_eval/run_eval.py`, `results/vlm_teacher_eval/full_eval/run_full_diqa_eval.py`, `research/vlm_calibration/evaluate_calibration.py`, and `research/correlation/ood_spread_analysis.py`. The torch `reset_parameters` monkeypatch pair is copied verbatim in `modal/benchmark_synthetic_ood.py:707-708` and `modal/run_deqa_ocr_iqa.py:271-272`.
- Recommendation: If more eval arms are coming, extract a small shared runner (arg parsing, env load, result-table writer) into one module the scripts import. If the experiments are done, accept the duplication.

## CQ-04 Type-ignore and noqa escape hatches (mostly justified)
- Severity: Low
- Effort: S
- CVE:
- Affected files: see evidence
- Evidence: 19 `# type: ignore` and 6 `# noqa`. The `type: ignore` cluster splits into deliberate test mutations of frozen dataclasses (`tests/uncertainty/test_model_normalizer.py:17`, `test_spread.py:81,109`), optional-import stubs in vendored Llamafactory (`data/loader.py:122-142`, `train/sft/trainer.py:78`), and dict-typing in research scripts. All carry a reason code (`[misc]`, `[arg-type]`, `[assignment]`). The 6 `noqa` are import-order (`E402`) after `sys.path` inserts and one `S301` on a self-owned pickle cache. None are bare suppressions.
- Recommendation: No action. These are localized and reason-coded.

## CQ-05 TODO/FIXME debt is small and old, concentrated in upstream
- Severity: Low
- Effort: S
- CVE:
- Affected files: see evidence
- Evidence: 14 markers total. 10 are in vendored upstream (`src/model/modeling_llama2.py` has 4 Flash-Attention TODOs; `src/model/modeling_mplug_owl2.py:122` "hacky fix for deepspeed zero3"; `Llamafactory/.../collator.py` 2 FIXMEs). The one actionable project marker, `src/train/train_mem.py:488` ("TODO I dont like auto resume << REMOVE IT"), dates to 2025-07-10 (oldest, via `git log -S`), about 10.5 months old. `modal/benchmark_synthetic_ood.py:1087` "TODO load from volume checkpoints" is the only marker in new code.
- Recommendation: Resolve or delete the `train_mem.py:488` auto-resume TODO (it gates commented-out code). Upstream markers stay.

## CQ-06 Assert-based validation in shipping code paths
- Severity: Low
- Effort: S
- CVE:
- Affected files: `DeQA-Score/src/datasets/pair_dataset.py:125`, `src/datasets/single_dataset.py:79`, `src/mm_utils.py:101`
- Evidence: `ruff --select S101` reports 88 asserts across new + upstream code. The 3 above are control-flow asserts in data/inference paths (`assert len(sources) == 1`), which `python -O` strips. Most of the other 85 are inside `tests/` (legitimate).
- Recommendation: For the 3 non-test asserts in data paths, raise explicit exceptions, since `python -O` strips asserts. Low risk in practice (training rarely uses `-O`).

## Clean areas (one line each)
- `DeQA-Score/src/uncertainty/` (the production pipeline): ruff clean, only 2 functions over complexity 10.
- Test count is healthy at 339 functions; only 2 skips, both gated on `torch not installed` (`tests/integration/test_contract.py:61,76`), no `xfail`.
- No `except: pass` swallow blocks (0 hits across the repo).
