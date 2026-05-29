# 07 Documentation and Developer Experience

Setup documentation is mostly accurate: every script the README and CLAUDE.md name exists (`train.sh`, `infer*.sh`, `eval_*.sh`, `scorer.py`, the `fig/singapore_flyer.jpg` sample), and the gradio removal is documented in the right places with no stale "run the demo" instructions left behind. Two concrete inaccuracies bite a new user: the documented quick-start import `from src import Scorer` raises `AttributeError` because `src/__init__.py` only exports the model class, and the documented required env var `GEMINI_API_KEY` is read nowhere in the code. The standing `PROJECT_REVIEW_REPORT.md` (2026-03-09) is a prior audit whose top finding (constants with no single source of truth) is still unresolved, so it reads as live but describes debt that persists.

## DOC-01 Documented quick-start import `from src import Scorer` is broken
- Severity: High
- Effort: S
- CVE:
- Affected files: `CLAUDE.md:42`, `DeQA-Score/README.md:103`, `DeQA-Score/src/__init__.py`
- Evidence: Both docs show `from src import Scorer`. `src/__init__.py` defines only a `__getattr__` that returns `MPLUGOwl2LlamaForCausalLM` and raises `AttributeError` for any other name. `Scorer` lives in `src/evaluate/scorer.py:14`. So the documented import fails; the working import is `from src.evaluate.scorer import Scorer`.
- Recommendation: Either re-export `Scorer` in `src/__init__.py` (`__getattr__` can handle the name), or fix both docs to the working path. The README copy is the upstream-facing one, so fix the import path there too.

## DOC-02 Documented required env var GEMINI_API_KEY is unused; ANTHROPIC_API_KEY is undocumented
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `CLAUDE.md:157`, `.env.example:1`
- Evidence: `CLAUDE.md:157` states the VLM scripts require `GEMINI_API_KEY` and `OPENROUTER_API_KEY`. `git grep GEMINI -- '*.py'` returns 0 hits. The code actually reads `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GCP_SA_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, and `ADOBE_CLIENT_ID/SECRET` (`git grep environ/getenv`). `ANTHROPIC_API_KEY` is used but not named in the CLAUDE.md requirement line; `GEMINI_API_KEY` is documented but dead.
- Recommendation: Update `CLAUDE.md` and `.env.example` to match the env vars the code reads. Remove `GEMINI_API_KEY` or wire it up; add `ANTHROPIC_API_KEY` and `GOOGLE_APPLICATION_CREDENTIALS` to the documented set.

## DOC-03 Prior review report reads as live but its top findings persist
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `PROJECT_REVIEW_REPORT.md`
- Evidence: Dated 2026-03-09, it claims "8 critical, 10 high" findings and names constants defined "independently in 5-7 locations with no single source of truth" as the top issue. This audit confirms that issue is still present (report 04, ARCH-01/ARCH-02, 2.5 months later). The document has no status column marking which findings are resolved.
- Recommendation: Add a resolution-status column or a header note dating each finding's state, or move it under `docs/audit/`. As a root-level undated-status doc it implies findings may be stale when they are not.

## DOC-04 Llamafactory patch-apply process is underdocumented
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `CLAUDE.md` (Qwen2.5-VL section), `Llamafactory/`
- Evidence: `CLAUDE.md` says the `Llamafactory/` files are "patches to be copied into a LLaMA-Factory installation (paths match)" but gives no target LLaMA-Factory version/commit and no copy/apply steps. A new user cannot reproduce the Qwen backend from this alone (see report 04, ARCH-05).
- Recommendation: Add the pinned LLaMA-Factory commit and a copy command (or a `make patch` target) to the Qwen section of `CLAUDE.md`.

## DOC-05 No ADRs for load-bearing decisions
- Severity: Low
- Effort: M
- CVE:
- Affected files: `docs/architecture/` (design docs, no ADRs)
- Evidence: `docs/architecture/` holds strategy/plan docs and PlantUML diagrams but no ADR records for decisions visible in code and config: the `pydantic<2` pin, the gradio removal, the two-model-backend split, the uncertainty-pipeline design. The gradio and pin rationale live in `CHANGELOG.md` and `docs/known-vulnerabilities.md`, which is partial.
- Recommendation: Add short ADRs for the pin policy and the two-backend split so the "why" survives independent of the changelog. Low priority.

## Clean areas (one line each)
- Script accuracy: all scripts named in README/CLAUDE.md exist under `DeQA-Score/scripts/`, and the sample image `fig/singapore_flyer.jpg` is present.
- No stale gradio instructions: the only `gradio` mentions in docs (`CHANGELOG.md:43`, `docs/known-vulnerabilities.md:19`) document its removal, correctly.
- The flash_attn separate-install gotcha is documented in CLAUDE.md's command section.
