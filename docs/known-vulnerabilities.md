---
schema_type: common
title: "Known Vulnerabilities"
status: published
owner: core-maintainer
purpose: "CVEs that cannot be immediately resolved in DeQA-Doc dependencies, with rationale and reassessment dates."
tags:
  - security
  - dependencies
---

> Tracks CVEs that cannot be immediately resolved. Review quarterly.
> No entry may age past 60 days without reassessment; escalate or resolve.
>
> Context: DeQA-Doc is a research fork of the DeQA-Score / mPLUG-Owl2
> implementation that won the VQualA 2025 DIQA Challenge. The model stack is
> pinned for reproducibility and championship-model compatibility. Most
> residuals below stem from those pins. The remediation pass on
> `fix/deps-vulnerability-remediation` removed the unused `gradio` web-demo
> stack and raised every transitive floor that could be raised without
> breaking a pin. `requires-python` is `>=3.10,<3.12` (the torch 2.0.1 cu118
> wheel range), so every transitive security floor applies unconditionally
> with no vulnerable Python 3.8/3.9 fallback.

Reassessment cadence: 2026-07-27 (60 days from documentation date 2026-05-28),
unless an earlier review is triggered by a release that loosens a pin.

---

## Category A: Pinned model stack (mPLUG-Owl2 compatibility) - DO NOT UPGRADE

These pins are mandatory for the championship mPLUG-Owl2 model. Upgrading any
of them breaks training/inference. They are documented as accepted residuals.

### transformers 4.36.1 (32 alerts)

| Field | Value |
| --- | --- |
| **Severity** | High (6), Medium (22), Low (4) |
| **Affected package** | transformers == 4.36.1 |
| **Patched version** | 4.38.0 / 4.48.0 / 4.50.0 / 4.51.0 / 4.52.1 / 4.53.0 / 5.0.0rc3 (all break mPLUG-Owl2) |
| **Representative advisories** | GHSA-wrfc-pvp9-mr9g, GHSA-qxrp-vhvm-j765, GHSA-hxxf-235m-72v3, CVE-2024-3568, CVE-2025-5197, CVE-2025-6921, CVE-2026-1839 (full set: 32 alerts) |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Exploitation scenario**: Most advisories are ReDoS or unsafe-deserialization
issues triggered by maliciously crafted model configs, tokenizer files, or
chat templates. In this project transformers only ever loads the project's own
trusted mPLUG-Owl2 / Qwen weights and tokenizers from controlled paths; no
untrusted model artifacts are loaded at runtime.

**Why deferred**: `transformers==4.36.1` is pinned for mPLUG-Owl2 compatibility
(custom `MPLUGOwl2LlamaForCausalLM` modeling code depends on the 4.36.x API).
Upgrading is a known-breaking change to the championship model and is out of
scope for dependency remediation (see CLAUDE.md "Org Standards Deviations").

**Compensating control**: Only trusted, project-owned model and tokenizer
artifacts are loaded. No public/untrusted model hub downloads in the
train/infer/eval paths.

**Planned resolution**: Tied to a future migration of the model stack off
mPLUG-Owl2 (e.g., consolidating on the Qwen2.5-VL backend, which can use a
modern transformers). No committed timeline.

### torch 2.0.1 + torchvision 0.15.2 (10 alerts)

| Field | Value |
| --- | --- |
| **Severity** | Critical (2), High (4), Medium (2), Low (2) |
| **Affected package** | torch == 2.0.1 (cu118) |
| **Patched version** | 2.2.0 / 2.5.0 / 2.6.0 / 2.7.0 / 2.7.1 / 2.8.0 / 2.9.0 (all break mPLUG-Owl2 + cu118 wheel set) |
| **Representative advisories** | GHSA-53q9-r3pm-6pq6 (critical), GHSA-pg7h-5qx3-wjr3, GHSA-5pcm-hx3q-hm94, PYSEC-2025-41, CVE-2025-3730 (full set: 10 alerts) |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Exploitation scenario**: Advisories cover unsafe deserialization of
malicious `.pt`/checkpoint files and local DoS via crafted tensors. Project
loads only its own trusted checkpoints; no untrusted `torch.load` of
attacker-supplied files.

**Why deferred**: `torch==2.0.1` / `torchvision==0.15.2` are pinned to the
PyTorch cu118 wheel index for mPLUG-Owl2 compatibility (and torch 2.0.1 is the
last with the required ABI for the pinned `flash_attn`/`deepspeed` builds).
Upgrading breaks the championship training pipeline.

**Compensating control**: Only trusted, project-owned checkpoints are loaded.
Training/inference run in controlled GPU environments, not exposed to untrusted
input.

**Planned resolution**: Tied to the same model-stack migration as transformers.
No committed timeline.

### deepspeed 0.14.5: PYSEC-2024-109 / CVE-2024-43497 (1 alert)

| Field | Value |
| --- | --- |
| **Severity** | High (remote code execution) |
| **Affected package** | deepspeed == 0.14.5 (train extra only) |
| **Patched version** | 0.15.1 (requires `pydantic>=2`, incompatible with the pinned `pydantic<2`) |
| **Advisory** | PYSEC-2024-109 / CVE-2024-43497 / GHSA-8cp5-3rf8-8gfh |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Exploitation scenario**: CVE-2024-43497 is an RCE reachable through crafted
DeepSpeed configuration input. In this project DeepSpeed is used only for
training, with project-authored config (`scripts/zero3.json` and the training
launch scripts); no attacker-supplied DeepSpeed config is ever loaded.

**Why deferred**: [#14](https://github.com/ByronWilliamsCPA/DeQA-Doc/pull/14)
bumped the pin to `deepspeed==0.15.1` (the first patched release), but DeepSpeed
0.15.x requires `pydantic>=2`, which conflicts with the mandatory `pydantic<2`
pin from the mPLUG-Owl2 stack. That bump made the `train` extra unsatisfiable
(`uv lock` fails), so the pin was lowered to `deepspeed==0.14.5` (the highest
release still compatible with `pydantic<2`). 0.14.5 predates the 0.15.1 fix, so
the advisory is an accepted residual. DeepSpeed is an optional, training-only
dependency.

**Compensating control**: DeepSpeed runs only in controlled GPU training
environments with project-owned configuration; it is not exposed to untrusted
input and is absent from the inference/eval paths. The Dependency Review CI
gates allow-list `GHSA-8cp5-3rf8-8gfh` so this tracked residual does not block
merges; the allow-list entry is removed when the pin moves to `>=0.15.1`.

**Planned resolution**: Fixed when the model stack migrates off `pydantic<2`
(tied to the same mPLUG-Owl2 -> Qwen2.5-VL migration as transformers/torch),
which unblocks `deepspeed>=0.15.1`. No committed timeline.

---

## Category B: Conservative pins (fixable in principle, deferred for model safety)

These are not in the hard pin list but were left unchanged to avoid subtle
behavioral changes to the championship model / its evaluation outputs. They are
lower-risk in this project's context.

### scikit-learn 1.2.2: PYSEC-2024-110 (2 alerts)

| Field | Value |
| --- | --- |
| **Severity** | Medium |
| **Affected package** | scikit-learn == 1.2.2 |
| **Patched version** | 1.5.0 (requires Python >=3.9) |
| **Advisory** | PYSEC-2024-110 / GHSA-jw8x-6495-233v |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Exploitation scenario**: The vulnerability is a sensitive-data leak via
`TfidfVectorizer.stop_words_`. This project does not use `TfidfVectorizer`
anywhere (scikit-learn is used for `StandardScaler`, KNN, and correlation
metrics in `research/ood_baselines`, `research/vlm_calibration`); the
vulnerable code path is unreachable here.

**Why deferred**: `scikit-learn==1.2.2` is an upstream `==` pin in the core
deps. Bumping to 1.5.0 (py3.9+) is technically possible but risks changing
numeric behavior in research OOD/calibration code that was validated against
1.2.2. The actual vulnerable feature is not used, so the real risk is
negligible.

**Compensating control**: `TfidfVectorizer` is not imported or used anywhere
in the codebase.

**Planned resolution**: Bump to >=1.5.0 the next time the research evaluation
code is revalidated, or drop py3.8 support.

### sentencepiece 0.1.99: CVE-2026-1260 (2 alerts)

| Field | Value |
| --- | --- |
| **Severity** | High |
| **Affected package** | sentencepiece == 0.1.99 |
| **Patched version** | 0.2.1 (requires Python >=3.9) |
| **Advisory** | CVE-2026-1260 / GHSA-38vq-g6vr-w8wf |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Exploitation scenario**: Crafted SentencePiece model file can trigger a
buffer issue. The project loads only its own trusted LLaMA/mPLUG-Owl2 tokenizer
model files.

**Why deferred**: `sentencepiece==0.1.99` is an upstream `==` pin and is
load-bearing for the LLaMA-2 tokenizer used by mPLUG-Owl2. Bumping to 0.2.1 is
a minor version change that could subtly alter tokenization behavior, which
would change model outputs. Deferred to protect championship-model
reproducibility.

**Compensating control**: Only trusted, project-owned tokenizer models are
loaded; no untrusted `.model` files.

**Planned resolution**: Validate 0.2.1 tokenization equivalence against 0.1.99,
then bump. Tied to the model-stack revalidation.

---

## Category C: Python 3.8 floor fallbacks (resolved: requires-python set to >=3.10,<3.12)

The project previously kept `requires-python = ">=3.8,<3.12"` and used
`python_full_version` markers to apply security floors only on newer Pythons,
leaving vulnerable fallbacks for Python 3.8/3.9. `requires-python` is now
`>=3.10,<3.12`: the narrowest range where every floor below resolves to its
patched release with no vulnerable fallback, while still inside the torch 2.0.1
cu118 wheel set (cp310/cp311). All packages in this table now unconditionally
resolve to their patched floors. No residuals remain from this category.

| Package | Former py3.8 fallback | Now-required floor | Advisory | Severity |
| --- | --- | --- | --- | --- |
| urllib3 | 2.2.3 | 2.7.0 | PYSEC-2026-141/142, GHSA-qccp-gfcp-xxvc | High/Medium |
| pillow | 10.4.0 | 12.2.0 | GHSA-whj4-6x5x-4v2j, GHSA-pwv6-vv43-88gr, et al. | High/Medium |
| requests | 2.32.x | 2.34.2 | CVE-2026-25645 / GHSA-gc5v-m9x4-r6x2 | Medium |
| pytest (dev) | 8.3.5 | 9.0.3 | CVE-2025-71176 / GHSA-6w46-j5rx-g56g | Medium |
| pygments | 2.19.2 | 2.20.0 | CVE-2026-4539 | Low |
| starlette | 0.44.0 | 0.49.3 | GHSA-7f5h-v6xp-fcq8, GHSA-2c2j-9gv5-cj73 | High/Medium |
| filelock | <3.20.1 | 3.20.1 | (advisory TBD) | Low |

**Resolved**: 2026-05-29 (`requires-python` set to `>=3.10,<3.12`; floors apply unconditionally).

### starlette PYSEC-2026-161 (forward-looking)

| Field | Value |
| --- | --- |
| **Severity** | Medium |
| **Affected package** | starlette <= 0.50.0 |
| **Patched version** | 1.0.1 |
| **Advisory** | PYSEC-2026-161 |
| **Date documented** | 2026-05-28 |
| **Reassessment due** | 2026-07-27 |

**Why deferred**: starlette 1.0.1 is gated by the pinned `fastapi`'s supported
starlette range; fastapi/uvicorn are themselves unused by any code path in this
fork (no `import fastapi`/`import uvicorn`). The 0.49/0.50 bump already cleared
the high/medium Dependabot alerts. The 1.0.1 fix will be picked up when fastapi
loosens its starlette ceiling.

**Compensating control**: fastapi/uvicorn are declared but not invoked by any
train/infer/eval code; no ASGI server is run.
