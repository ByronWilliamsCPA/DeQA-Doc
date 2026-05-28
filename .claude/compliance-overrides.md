# Compliance Overrides Registry

This file is a per-repository compliance override registry consumed by the
`/repo-audit` workflow (the `repo-compliance` skill). Each entry suppresses a
specific manifest check ID for a documented, intentional reason. When the audit
runs, any check listed here is logged as `OVERRIDE` with the stated reason
instead of being re-raised as a finding.

These overrides are not permanent waivers. Every entry carries an `expires`
review date; once that date passes, the override should be re-evaluated and
either renewed with fresh justification or removed so the check re-activates.

This file is the machine-readable companion to the "Org Standards Deviations"
and "CI/CD" sections of the repository `CLAUDE.md`. Those prose sections are the
human-readable explanation; this registry is what the audit tooling parses.
Keep the two in sync: if you change a deviation in `CLAUDE.md`, update the
corresponding entry here, and vice versa.

## Repository Profile

- **Type**: research fork / paper implementation (adapts DeQA-Score to Document
  Image Quality Assessment; VQualA 2025 DIQA Challenge entry)
- **Visibility**: public
- **Primary maintainer**: Byron Williams
- **Code populations**: this repository contains two distinct code populations
  that are held to different standards:
  1. **Upstream / vendored code**: `DeQA-Score/src/` and `Llamafactory/`. This
     is forked from an upstream paper implementation, uses its own style, carries
     Chinese-language comments, and lacks type annotations and docstrings.
     It is intentionally exempt from strict org tooling.
  2. **New project code**: `results/` and `DeQA-Score/src/uncertainty/`. New
     modules follow org standards (conventional commits, type hints, docstrings,
     Ruff formatting) and are the only paths that should be strictly checked.

## Active Overrides

| Check ID | Reason (summary) | Scope | Expires |
| --- | --- | --- | --- |
| TOOL-001 | Upstream uses its own style and Chinese comments; Ruff scoped to new code via pre-commit | Upstream paths | 2026-11-28 |
| TOOL-002 | No type annotations on legacy upstream code; strict BasedPyright would require extensive refactoring | Repo-wide for upstream; new code adopts gradually | 2026-11-28 |
| TOOL-003 | Upstream lacks docstrings; darglint not meaningful on vendored code | Upstream paths | 2026-11-28 |
| TOOL-004 | Same as TOOL-003: upstream lacks docstrings; interrogate coverage not meaningful | Upstream paths | 2026-11-28 |
| CI-003 | Research project, no release cadence, not distributing artifacts; OpenSSF Scorecard workflow not warranted | Repo-wide | 2026-11-28 |
| FOUND-GOVERNANCE | Single-maintainer research fork; org-level health files in ByronWilliamsCPA/.github cover governance | Repo-wide | 2026-11-28 |
| FOUND-CODEOWNERS | Solo maintainer; no multi-owner review routing needed | Repo-wide | 2026-11-28 |
| DEP-PINS | Legacy pins required for mPLUG-Owl2 compatibility; must not bump without full training-pipeline re-validation | Dependency manifests | 2026-11-28 |
| GIT-SIGNED-COMMITS | Research collaboration repo with external contributors; signed commits not required | Repo-wide | 2026-11-28 |

```yaml
overrides:
  - check: TOOL-001
    name: Ruff strict rule set
    reason: >-
      Upstream code (DeQA-Score/src/, Llamafactory/) uses its own style and
      carries Chinese-language comments. Ruff is scoped to new code only via
      pre-commit and CI; full enforcement would require rewriting vendored code.
    scope: upstream paths (DeQA-Score/src/, Llamafactory/)
    expires: 2026-11-28

  - check: TOOL-002
    name: BasedPyright strict config
    reason: >-
      Legacy upstream code has no type annotations. Strict typing would require
      extensive refactoring of vendored code. New code in results/ and
      DeQA-Score/src/uncertainty/ may adopt type hints gradually.
    scope: repo-wide for upstream; new code adds gradually
    expires: 2026-11-28

  - check: TOOL-003
    name: darglint / docstring linting
    reason: Upstream code lacks docstrings; darglint is not meaningful on vendored code.
    scope: upstream paths (DeQA-Score/src/, Llamafactory/)
    expires: 2026-11-28

  - check: TOOL-004
    name: interrogate docstring coverage
    reason: >-
      Same reason as TOOL-003: upstream code lacks docstrings, so docstring
      coverage thresholds do not apply to vendored paths.
    scope: upstream paths (DeQA-Score/src/, Llamafactory/)
    expires: 2026-11-28

  - check: CI-003
    name: OpenSSF Scorecard workflow
    reason: >-
      Research project with no release cadence and no distributed artifacts. A
      Scorecard supply-chain workflow is not warranted for this repo.
    scope: repo-wide
    expires: 2026-11-28

  - check: FOUND-GOVERNANCE
    name: GOVERNANCE.md presence
    reason: >-
      Single-maintainer research fork. Org-level community health files in
      ByronWilliamsCPA/.github provide org-wide governance, so a repo-local
      GOVERNANCE.md is redundant.
    scope: repo-wide
    expires: 2026-11-28

  - check: FOUND-CODEOWNERS
    name: CODEOWNERS presence
    reason: >-
      Solo maintainer; there is no multi-owner review routing to encode, so a
      CODEOWNERS file would add no value.
    scope: repo-wide
    expires: 2026-11-28

  - check: DEP-PINS
    name: Dependency-freshness / version-currency
    reason: >-
      Pinned legacy dependencies (torch==2.0.1, transformers==4.36.1,
      accelerate==0.21.0, peft==0.4.0) are required for mPLUG-Owl2
      compatibility. These pins must not be bumped without full
      training-pipeline re-validation, so version-currency findings are
      overridden as a deliberate policy choice.
    scope: dependency manifests (pyproject.toml, uv.lock)
    expires: 2026-11-28

  - check: GIT-SIGNED-COMMITS
    name: Signed-commit requirement
    reason: >-
      Research collaboration repo with external contributors; signed commits
      are not required here, deviating from the org default.
    scope: repo-wide
    expires: 2026-11-28
```

## Known Gaps (not overridden)

The following items are NOT excused. They represent real gaps that should be
fixed rather than suppressed. They are recorded here so the audit does not
mistake them for accepted overrides.

- **TOOL-005: pip-audit / dependency vulnerability scanning is absent.** No
  dependency vulnerability scanning currently runs. Recommendation: add
  `uv run pip-audit` to CI and/or pre-commit so dependency CVEs are surfaced.
  Note that the legacy pins (see DEP-PINS) may produce findings that cannot be
  bumped; those should be triaged and documented in
  `docs/known-vulnerabilities.md` rather than ignored.
- **SECURITY.md content not verified.** A SECURITY.md may exist, but its content
  has not been verified against the OpenSSF baseline. Recommendation: confirm
  it contains a real vulnerability-reporting process and supported-version
  policy.
- **CI lint steps are warnings-only.** Current CI lint steps emit
  `|| echo ::warning::` rather than blocking. New code in `results/` and
  `DeQA-Score/src/uncertainty/` is held to org standards, so once those paths
  are lint-clean, make their lint steps blocking instead of advisory.

## How to Use

When `/repo-audit` runs against this repository, each check ID listed under
**Active Overrides** is logged as `OVERRIDE` with the stated reason instead of
being raised as a finding. Items under **Known Gaps** are intentionally left
un-overridden so they continue to appear as findings until fixed.

To re-enable a check, delete its entry from both the table and the YAML block
above; the next audit run will then evaluate that check normally. To extend an
override past its review date, update the `expires` field with a fresh
justification. Keep this registry aligned with the "Org Standards Deviations"
section of `CLAUDE.md`.
