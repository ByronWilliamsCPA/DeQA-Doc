# 00 Final Report: Holistic Legacy and Architecture Audit

Date (UTC): 2026-05-29. HEAD: `c7e7593`. Branch: `claude/repo-audit-3HAXc`.

## Repo map (Phase 0)

- Primary language: Python. 197 tracked `.py` files, 50,918 lines. 469 tracked files total (82 `.md`, 59 `.json`, 26 `.jsonl`, 42 `.png`, 13 `.pdf`).
- Build/packaging: `uv` with `DeQA-Score/pyproject.toml` (setuptools backend) and `DeQA-Score/uv.lock` (92 packages). A second project, `research/ocr_iqa_correlation/pyproject.toml`, has no lockfile. No root `pyproject.toml`.
- Runtime: Python 3.11.15 in this environment; `requires-python = ">=3.10,<3.12"`. Pinned for the mPLUG-Owl2 stack: `torch==2.0.1` (2023-05), `transformers==4.36.1` (2023-12), CUDA 11.8 wheels.
- Tests: pytest, 339 test functions under `DeQA-Score/tests/` (proper suites in `uncertainty/`, `expansion/`, `integration/`; `model/` are exploratory scripts). Torch/CUDA tests are not run in CI.
- CI: 10 GitHub Actions workflows. Static analysis: ruff (pre-commit on `src/uncertainty/`, CI on `results/`), bandit, `.qlty/qlty.toml` (9 plugins), `.pre-commit-config.yaml`, renovate.
- Size/churn: largest files are experiment scripts (`modal/benchmark_synthetic_ood.py` 1164, `research/threshold_sensitivity/run_sweep.py` 1035) and vendored model code (`src/model/visual_encoder.py` 1017). Most-churned: `Readme.md` (10), `DeQA-Score/pyproject.toml` (8), `.github/workflows/ci.yml` (8), `src/uncertainty/ood_wrapper.py` (6), `src/train/loss.py` (6). 102 commits, 2025-07-10 to 2026-05-28.
- Migration residue: none. No `requirements*.txt`/`setup.py`/`poetry.lock`/`Pipfile`. Clean uv migration.

The codebase has three strata: vendored upstream (mPLUG-Owl2 model code, Llamafactory patches), a maintained production pipeline (`DeQA-Score/src/uncertainty/` plus core train/eval), and a large satellite tier of one-off experiment scripts (`research/`, `results/`, `modal/`). The audit ran all seven domains; each is sized to the relevant stratum.

## Code quality: critical analysis

The production pipeline is the strong part. `DeQA-Score/src/uncertainty/` is ruff-clean, has only 2 functions over complexity 10, reason-codes all its type suppressions, and is backed by 339 tests. If the whole repo were held to that bar, this would be a healthy codebase.

It is not held to that bar, and the weakness is measurement, not the code. There is no coverage figure for the repo: `coverage.yml` and `qlty.yml` both upload a `coverage.xml` artifact that CI never produces, so 339 tests run to no recorded number (CQ-01). The experiment tier carries 37 of 40 high-complexity functions, with `main()` bodies reaching cyclomatic complexity 30, and copy-pasted harness scaffolding across `research`/`results`/`modal` (CQ-02, CQ-03). That tier is write-once analysis code, so the right call is to stop linting it harder and instead wall it off from the production bar, which the configs already half-do. Debt markers are few (14 TODO/FIXME, mostly upstream) and the oldest actionable one dates to 2025-07-10.

## Architecture: critical analysis

The core package is well-layered: `src/constants.py` is a genuine single source of truth, imported by 10 core modules. The structure works against maintainers at the seams. The same five quality-level names are re-declared verbatim in at least 7 modules outside `constants.py` instead of imported (ARCH-01), and the third dimension is spelled two incompatible ways, `color` in the pipeline and `color_fidelity` in the papers tier (ARCH-02). That is not cosmetic: a dict keyed on one spelling silently drops data keyed on the other. The satellite tiers reach the core through 35 `sys.path` insertions rather than a package install (ARCH-03), which is why they copy constants instead of importing them; fixing the install fixes both. The Qwen backend is a set of patch files meant to be copied into a separate LLaMA-Factory install with no pinned target commit (ARCH-05), so that half of the "two backends" claim is not reproducible from the repo.

The prior `PROJECT_REVIEW_REPORT.md` (2026-03-09) named the constants-duplication issue as its top finding. It is still here 2.5 months later. That is the single clearest signal of architectural drift: a known, named, top-priority issue that did not move.

## Cross-cutting themes (no single subagent owned these)

1. Gates that do not gate. The production code is linted only by a local pre-commit hook that never runs in CI (CICD-02); CI's own ruff/bandit steps are advisory `|| echo` warnings (CICD-03); the coverage pipeline measures nothing (CQ-01); the compatibility matrix points at a `src/`/`tests/` layout this repo does not have and tests Python versions it cannot support (CICD-01). Across domains, the enforcement scaffolding exists but is wired to no-ops. This is the dominant root cause and it spans CI/CD, code-quality, and docs.

2. Copy instead of import. The constant duplication (ARCH-01/02), the `sys.path` hacks (ARCH-03), and the duplicated experiment harnesses (CQ-03) are the same habit at different scales: the satellite tier was built by copying from the core rather than depending on it. One package install plus shared-constant imports retires three findings.

3. Age stratification by design, with manual upkeep. The pinned 2023-era model stack (DEP-01/02) is deliberate and documented, but the consequences are hand-maintained: a manual transitive-floor list in `pyproject.toml` (DEP-04) and a manual CVE register (SEC-02). These work today and rot silently; they need an automated scanner behind them, not more manual edits.

4. Documentation that describes intent, not state. `from src import Scorer` is documented in two files and raises `AttributeError` (DOC-01); a required env var `GEMINI_API_KEY` is read nowhere (DOC-02); a prior review reads as current while its findings stand (DOC-03). The docs were written to the design and not re-checked against the code.

Where subagents overlapped, I resolved as follows. The coverage gap surfaced in both code-quality and CI/CD; I assigned the finding to CQ-01 (measurement) and kept CICD-04 narrowly on the redundant uploader workflows. The GitHub Actions posture split cleanly: security (report 05) confirmed SHA-pinning and least-privilege permissions are sound, while CI/CD (report 06) owns the versioning/gating defects. No direct contradictions between reports.

## Prioritized remediation backlog

Sorted by severity (High, Medium, Low) then effort (S, M, L).

| ID | Finding | Domain | Severity | Effort | Files |
| --- | --- | --- | --- | --- | --- |
| CICD-01 | Compat matrix points at nonexistent `src/`/`tests/` and unsupported Python | cicd | High | S | .github/workflows/python-compatibility.yml |
| CICD-02 | Production code linted only by local pre-commit, never in CI | cicd | High | S | .pre-commit-config.yaml; .github/workflows/ci.yml |
| DOC-01 | Documented `from src import Scorer` raises AttributeError | docs | High | S | CLAUDE.md; DeQA-Score/README.md; DeQA-Score/src/__init__.py |
| ARCH-02 | Dimension name split: `color` vs `color_fidelity` | architecture | High | S | research/papers/shared/constants.py; DeQA-Score/src/uncertainty/pseudo_label.py |
| ARCH-01 | Quality-level constants re-declared in 9+ modules | architecture | High | M | DeQA-Score/src/uncertainty/*; modal/*; research/* |
| CQ-01 | No real coverage measurement despite a coverage pipeline | code-quality | High | M | .github/workflows/coverage.yml; .github/workflows/ci.yml |
| DEP-01 | Core ML stack pinned to 2-3 year old releases | dependencies | High | L | DeQA-Score/pyproject.toml |
| DEP-02 | torch 2.0.1 caps runtime at Python 3.11 / CUDA 11.8 | dependencies | High | L | DeQA-Score/pyproject.toml |
| LEG-01 | Vendored transformers internals coupled to the 4.36.1 pin | legacy-code | High | L | DeQA-Score/src/model/modeling_llama2.py; modeling_attn_mask_utils.py |
| DEP-03 | research/ocr_iqa_correlation has no lockfile | dependencies | Medium | S | research/ocr_iqa_correlation/pyproject.toml |
| LEG-02 | sys.path.insert import hack in model loader | legacy-code | Medium | S | DeQA-Score/src/model/modeling_llama2.py |
| SEC-01 | No git-history secret scan and no detect-secrets baseline | security | Medium | S | .pre-commit-config.yaml; .github/workflows/ci.yml |
| CICD-03 | CI ruff/bandit gates are advisory-only | cicd | Medium | S | .github/workflows/ci.yml |
| CICD-04 | Redundant coverage uploaders consuming a missing artifact | cicd | Medium | S | .github/workflows/coverage.yml; .github/workflows/qlty.yml |
| CICD-05 | Contradictory CodeQL setup instructions | cicd | Medium | S | .github/workflows/ci.yml; .github/workflows/codeql.yml |
| CICD-06 | Mutable `@main` reference for a reusable workflow | cicd | Medium | S | .github/workflows/qlty.yml; .github/workflows/coverage.yml |
| DOC-02 | Documented `GEMINI_API_KEY` unused; `ANTHROPIC_API_KEY` undocumented | docs | Medium | S | CLAUDE.md; .env.example |
| DOC-03 | Prior review report reads as live but findings persist | docs | Medium | S | PROJECT_REVIEW_REPORT.md |
| DOC-04 | Llamafactory patch-apply process underdocumented | docs | Medium | S | CLAUDE.md; Llamafactory/ |
| DEP-04 | Manual transitive-floor list with no automated check | dependencies | Medium | M | DeQA-Score/pyproject.toml |
| CQ-03 | Duplicated experiment-harness scaffolding | code-quality | Medium | M | research/**; results/**; modal/** |
| ARCH-03 | Satellite tiers reach the core via 35 sys.path inserts | architecture | Medium | M | research/**; results/**; modal/** |
| ARCH-05 | Qwen backend relies on copy-into-install patching, no pinned commit | architecture | Medium | M | Llamafactory/src/llamafactory/** |
| CQ-02 | High-complexity functions in experiment scripts (max 30) | code-quality | Medium | L | modal/**; results/**; research/** |
| SEC-02 | Pinned dependencies with open CVEs (trusted-input control) | security | Medium | L | DeQA-Score/pyproject.toml; docs/known-vulnerabilities.md |
| DEP-05 | No SBOM generated | dependencies | Low | S | .github/workflows/ |
| LEG-04 | os.path where pathlib fits (new code) | legacy-code | Low | S | modal/*; DeQA-Score/src/uncertainty/* |
| LEG-05 | Bare except clauses in upstream paths | legacy-code | Low | S | DeQA-Score/src/datasets/utils.py; src/evaluate/iqa_eval.py |
| LEG-06 | .format() residue in new code | legacy-code | Low | S | research/threshold_sensitivity/run_sweep.py; results/vlm_teacher_eval/prompts.py |
| LEG-07 | pydantic v1 API (deliberate, documented) | legacy-code | Low | S | DeQA-Score/src/uncertainty/metadata_schema.py; metadata_io.py |
| CQ-04 | type-ignore/noqa escape hatches (justified) | code-quality | Low | S | DeQA-Score/tests/uncertainty/*; Llamafactory/src/llamafactory/data/loader.py |
| CQ-05 | TODO/FIXME debt small and old | code-quality | Low | S | DeQA-Score/src/train/train_mem.py; modal/benchmark_synthetic_ood.py |
| CQ-06 | Assert-based validation in data/inference paths | code-quality | Low | S | DeQA-Score/src/datasets/pair_dataset.py; src/mm_utils.py |
| SEC-03 | eval() on a path component in weight conversion | security | Low | S | DeQA-Score/src/model/convert_mplug_owl2_weight_to_hf.py |
| SEC-04 | requests calls without timeout | security | Low | S | DeQA-Score/src/evaluate/iqa_eval.py; eval_qbench_mcq.py; iqa_eval_qwen.py |
| SEC-05 | pickle.load on a cache file | security | Low | S | research/vlm_calibration/evaluate_calibration.py |
| CICD-07 | harden-runner pinned to two different versions | cicd | Low | S | .github/workflows/ci.yml; codeql.yml; security-analysis.yml |
| LEG-03 | typing.List/Dict generics instead of builtins (new code) | legacy-code | Low | M | DeQA-Score/src/uncertainty/* |
| ARCH-04 | research/ and results/ overlap in purpose | architecture | Low | M | research/; results/ |
| SEC-06 | Broad except Exception handlers (43 sites) | security | Low | M | repo-wide *.py |
| DOC-05 | No ADRs for load-bearing decisions | docs | Low | M | docs/architecture/ |

41 findings: 9 High, 16 Medium, 16 Low. No Critical: no live secret, no reachable RCE on untrusted input, no broken lockfile, no imminent runtime break.

## Verdict

Drifting, not at-risk. The core engineering is sound (clean production pipeline, real test suite, sound security hygiene, deliberate and documented dependency pins). What has drifted is the gap between the controls the repo declares and the controls it enforces, plus a known architectural issue that has sat unresolved for 2.5 months. Nothing here threatens correctness or supply chain today; left alone, the no-op gates will let real regressions through later.

The three changes that move it most:

1. Make the declared gates real: run pre-commit (or scoped ruff) in CI as a blocking check, fix the compatibility matrix to this repo's layout, and either wire up coverage or delete the pipeline that pretends to measure it (CICD-01, CICD-02, CQ-01).
2. Retire copy-instead-of-import: install `DeQA-Score` as a package in the satellite environments, then import `LEVEL_NAMES`/`DIMENSIONS` from `src.constants` everywhere and reconcile `color` vs `color_fidelity` (ARCH-01, ARCH-02, ARCH-03).
3. Fix the two documented-but-broken developer entry points: the `from src import Scorer` import and the `GEMINI_API_KEY` env var (DOC-01, DOC-02).
