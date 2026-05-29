# 06 CI/CD and Tooling

Ten workflows, all third-party actions SHA-pinned, per-job least-privilege permissions: the security hygiene is good. The problems are gates that do not gate. The production code (`DeQA-Score/src/uncertainty/`) is linted only by a local pre-commit hook that never runs in CI, so its quality bar is bypassable. CI's own ruff/bandit steps target `results/` and are advisory-only (`|| echo "::warning::"`), so they never fail a build. The Python compatibility matrix is misconfigured for this repo's layout (it points at a top-level `src/`/`tests/` that does not exist) and tests Python 3.12/3.13 that the project cannot support. Two workflows upload coverage that CI never produces. And `ci.yml` and `codeql.yml` carry contradictory instructions about whether CodeQL default setup is on.

## Lint / scan coverage matrix (path x enforcement)

| Path | pre-commit ruff (local) | CI ruff | CI bandit | qlty config | Enforced-in-CI gate |
| --- | --- | --- | --- | --- | --- |
| `DeQA-Score/src/uncertainty/` (production) | yes | no | no | included | none |
| `results/` | no | yes (advisory) | yes (advisory) | excluded | none (warnings only) |
| `research/`, `modal/` | no | no | no | excluded | none |
| upstream `src/model,train,...`, `Llamafactory/` | no | no | no | excluded | none (intended) |

## CICD-01 Python compatibility matrix points at nonexistent paths and unsupported versions
- Severity: High
- Effort: S
- CVE:
- Affected files: `.github/workflows/python-compatibility.yml:9-21,40-46`
- Evidence: The workflow filters on `paths: src/**/*.py` and `tests/**/*.py` and passes `source-directory: 'src'` with `test-command: 'pytest tests/ ...'`, but the repo has no top-level `src/` or `tests/` (they live under `DeQA-Score/`). So push/PR runs never trigger (paths never match), and when the weekly cron fires it finds nothing to test. It also sets `python-versions: ["3.10","3.11","3.12","3.13"]` while `DeQA-Score/pyproject.toml:11` pins `>=3.10,<3.12` and torch 2.0.1 has no 3.12/3.13 wheels.
- Recommendation: Point `source-directory`, the path filters, and the test command at `DeQA-Score/`, and drop 3.12/3.13 from the matrix until the torch upgrade lands. As written this workflow gives false assurance.

## CICD-02 Production code is not linted in CI (pre-commit only, never run in CI)
- Severity: High
- Effort: S
- CVE:
- Affected files: `.pre-commit-config.yaml:79-88`, `.github/workflows/ci.yml` (ruff steps scoped to `results/`)
- Evidence: ruff lint/format on `DeQA-Score/src/uncertainty/` exists only in `.pre-commit-config.yaml` (`files: ^DeQA-Score/src/uncertainty/`). No workflow runs `pre-commit` (`grep -rln pre-commit .github/workflows/` is empty). CI's ruff steps target `results/` only. A contributor who skips the local hook lands unlinted production code.
- Recommendation: Add a CI job that runs `pre-commit run --all-files` (or a ruff step scoped to `DeQA-Score/src/uncertainty/`) as a blocking check.

## CICD-03 CI quality gates are advisory-only
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `.github/workflows/ci.yml` (Ruff format check, Ruff lint, Bandit steps)
- Evidence: The ruff-format, ruff-lint, and bandit steps all end in `|| echo "::warning::..."`, so a failure becomes a warning annotation and the job still passes. Only the hardcoded-secret grep and `.env`-tracked check can fail the build.
- Recommendation: Drop the `|| echo` fallback on at least the ruff lint step so style/lint regressions block merge, or accept that these are informational and say so in the workflow name.

## CICD-04 Redundant coverage uploaders consuming an artifact CI never builds
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `.github/workflows/coverage.yml:18-33`, `.github/workflows/qlty.yml:13-27`
- Evidence: Both `coverage.yml` and `qlty.yml` trigger on `workflow_run: ["CI"] completed` and both call `ByronWilliamsCPA/.github/.../python-qlty-coverage.yml@main` with `coverage-artifact-name: coverage-reports` / `coverage.xml`. CI produces no such artifact (see report 03, CQ-01). Two workflows do the same no-op.
- Recommendation: Delete one of the two, and make the survivor depend on a CI step that actually emits `coverage.xml`. Resolve jointly with CQ-01.

## CICD-05 Contradictory CodeQL setup instructions
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `.github/workflows/ci.yml` (CodeQL comment block), `.github/workflows/codeql.yml:6,1-5`
- Evidence: `ci.yml` states "CodeQL security analysis is handled by GitHub's default setup. Do not add an advanced CodeQL configuration here; it conflicts with the default setup and causes SARIF upload failures." But `codeql.yml` is exactly an advanced configuration (`codeql-action/init` + `analyze`, `queries: security-extended`) and its own header says "GitHub's CodeQL default setup must remain DISABLED." If default setup is enabled, the two collide on SARIF upload; the comments cannot both be acted on.
- Recommendation: Confirm default setup is disabled (codeql.yml is the active path) and delete the stale contradicting comment in `ci.yml`. One source of truth for CodeQL.

## CICD-06 Mutable @main reference for a reusable workflow
- Severity: Medium
- Effort: S
- CVE:
- Affected files: `.github/workflows/qlty.yml:14`, `.github/workflows/coverage.yml:26`
- Evidence: Both reference `ByronWilliamsCPA/.github/.github/workflows/python-qlty-coverage.yml@main` (mutable branch), while `python-compatibility.yml:39`, `security-analysis.yml`, and `scorecard.yml` pin their org reusable workflows to a commit SHA (`@c22009cc...`, `@d18c9304...`). The two `@main` references can change behavior without a repo change.
- Recommendation: Pin the qlty/coverage reusable workflow to a SHA like the others, and let Renovate bump it.

## CICD-07 harden-runner pinned to two different versions
- Severity: Low
- Effort: S
- CVE:
- Affected files: `.github/workflows/ci.yml`, `reuse.yml` (`@a5ad31d6 # v2.19.1`) vs `codeql.yml`, `security-analysis.yml` (`@91182ccc # v2.10.1`)
- Evidence: `step-security/harden-runner` appears at SHA `a5ad31d6` (v2.19.1) in some workflows and `91182ccc` (v2.10.1) in others. Version drift across a security action.
- Recommendation: Align all `harden-runner` uses to one SHA; Renovate's "Group GitHub Actions" rule should keep them together.

## Clean areas (one line each)
- No deprecated Actions commands: 0 hits for `set-output`/`save-state`/`::set-`; all major actions are current (checkout v6, codeql v4, upload-artifact v7, setup-uv v8).
- All third-party actions are SHA-pinned; uv caching is enabled in CI (`setup-uv` with `enable-cache: true`).
- Per-job `permissions:` are least-privilege across the workflows (`contents: read` default, `security-events: write` only where CodeQL needs it).
