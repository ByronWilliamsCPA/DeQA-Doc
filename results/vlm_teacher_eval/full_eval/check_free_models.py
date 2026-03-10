"""Daily check for new free vision models on OpenRouter.

Queries the OpenRouter /models API for models ending in :free that accept
image input, compares against existing checkpoints, and optionally launches
evaluation for new/incomplete models.

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/check_free_models.py

    # Check only (dry run — default):
    ... check_free_models.py

    # Auto-evaluate new/incomplete models:
    ... check_free_models.py --run

    # Limit images per model (for testing):
    ... check_free_models.py --run --limit 10

    # Show all free vision models including already evaluated:
    ... check_free_models.py --all

    # JSON output (for automation / cron):
    ... check_free_models.py --json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Path setup — same as run_full_diqa_eval.py
EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from results.vlm_teacher_eval.full_eval.run_full_diqa_eval import (
    CHECKPOINT_DIR,
    ImageResult,
    append_checkpoint,
    compute_all_metrics,
    download_test_data,
    evaluate_model,
    load_checkpoint,
    load_env,
    load_ground_truth,
)
from results.vlm_teacher_eval.prompts import USER_PROMPT, build_system_prompt

STATE_FILE = EVAL_DIR / "free_models_state.json"

# Thresholds
MIN_COMPLETE = 950


@dataclass(frozen=True)
class FreeModel:
    """A free vision model from OpenRouter."""

    model_id: str
    name: str
    base_model_id: str
    context_length: int
    max_completion_tokens: int | None


def fetch_free_vision_models(api_key: str) -> list[FreeModel]:
    """Query OpenRouter API for free models with vision capabilities."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    models: list[FreeModel] = []
    for m in data.get("data", []):
        model_id: str = m.get("id", "")
        if not model_id.endswith(":free"):
            continue

        arch = m.get("architecture", {})
        input_modalities = [x.lower() for x in arch.get("input_modalities", [])]
        modality = arch.get("modality", "").lower()

        has_image = (
            "image" in input_modalities
            or "image" in modality
            or "multimodal" in modality
        )
        if not has_image:
            continue

        base_id = model_id.removesuffix(":free")
        top_provider = m.get("top_provider", {})

        models.append(
            FreeModel(
                model_id=model_id,
                name=m.get("name", ""),
                base_model_id=base_id,
                context_length=m.get("context_length", 0),
                max_completion_tokens=top_provider.get("max_completion_tokens"),
            )
        )

    return sorted(models, key=lambda x: x.model_id)


def checkpoint_line_count(base_model_id: str) -> int:
    """Count valid (non-error) lines in checkpoint file."""
    safe_name = base_model_id.replace("/", "__")
    cp = CHECKPOINT_DIR / f"{safe_name}.jsonl"
    if not cp.exists():
        return 0
    count = 0
    for line in cp.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not item.get("error"):
                count += 1
        except json.JSONDecodeError:
            continue
    return count


def load_state() -> dict[str, Any]:
    """Load previous state for diff detection."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"known_models": [], "last_check": None}


def save_state(free_models: list[FreeModel]) -> None:
    """Save current state for future diff detection."""
    state = {
        "known_models": [m.model_id for m in free_models],
        "last_check": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def run_eval_for_model(
    base_model_id: str,
    ground_truth: list,
    api_key: str,
    system_prompt: str,
    gt_lookup: dict,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run evaluation for a single model and return metrics."""
    results = evaluate_model(
        model_id=base_model_id,
        ground_truth=ground_truth,
        api_key=api_key,
        system_prompt=system_prompt,
        limit=limit,
    )

    ok = sum(1 for r in results if not r.error)
    err = sum(1 for r in results if r.error)
    print(f"\n  Results: {ok} success, {err} errors")

    metrics = compute_all_metrics(results, gt_lookup)

    if "overall_srcc" in metrics:
        print(f"  Overall SRCC: {metrics['overall_srcc']:.4f}")
        print(f"  Sharpness SRCC: {metrics.get('sharpness_srcc', 'N/A')}")
        print(f"  Color SRCC: {metrics.get('color_srcc', 'N/A')}")
        print(f"  wSRCC: {metrics.get('wsrcc', 'N/A')}")

    return metrics


def main() -> None:
    """Check for new free vision models and optionally evaluate them."""
    parser = argparse.ArgumentParser(
        description="Check for new free vision models on OpenRouter"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Auto-launch evaluation for new/incomplete models",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit images per model (for testing). Default: all 1000",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all free vision models including already evaluated",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for automation)",
    )
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set in environment or .env")
        sys.exit(1)

    # Fetch current free vision models
    print("Querying OpenRouter for free vision models...")
    free_models = fetch_free_vision_models(api_key)
    print(f"Found {len(free_models)} free vision model(s)\n")

    # Load previous state for diff
    prev_state = load_state()
    prev_known = set(prev_state.get("known_models", []))
    last_check = prev_state.get("last_check", "never")

    # Categorize models
    new_models: list[FreeModel] = []
    in_progress: list[tuple[FreeModel, int]] = []
    complete: list[tuple[FreeModel, int]] = []
    removed = prev_known - {m.model_id for m in free_models}

    for m in free_models:
        n_done = checkpoint_line_count(m.base_model_id)
        if n_done >= MIN_COMPLETE:
            complete.append((m, n_done))
        elif n_done >= 1:
            in_progress.append((m, n_done))
        else:
            new_models.append(m)

    # JSON output mode
    if args.json:
        result = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "previous_check": last_check,
            "free_vision_models": [
                {"id": m.model_id, "name": m.name, "base_id": m.base_model_id}
                for m in free_models
            ],
            "new": [m.model_id for m in new_models],
            "in_progress": [
                {"model": m.model_id, "images_done": n} for m, n in in_progress
            ],
            "complete": [
                {"model": m.model_id, "images_done": n} for m, n in complete
            ],
            "removed": sorted(removed),
        }
        print(json.dumps(result, indent=2))
        save_state(free_models)
        return

    # Human-readable output
    print(f"Last check: {last_check}")
    print("=" * 70)

    if removed:
        print(f"\n  REMOVED since last check ({len(removed)}):")
        for model_id in sorted(removed):
            print(f"    - {model_id}")

    newly_appeared = {m.model_id for m in free_models} - prev_known
    if newly_appeared and prev_known:
        print(f"\n  NEWLY APPEARED ({len(newly_appeared)}):")
        for model_id in sorted(newly_appeared):
            print(f"    + {model_id}")

    if new_models:
        print(f"\n  NEW — no checkpoint ({len(new_models)}):")
        for m in new_models:
            print(f"    {m.model_id}")
            print(f"      {m.name} | ctx={m.context_length}")
    else:
        print("\n  No new unevaluated models.")

    if in_progress:
        print(f"\n  IN PROGRESS ({len(in_progress)}):")
        for m, n in in_progress:
            pct = n / 10  # percent out of 1000
            print(f"    {m.base_model_id}: {n}/1000 ({pct:.1f}%)")

    if args.all and complete:
        print(f"\n  COMPLETE ({len(complete)}):")
        for m, n in complete:
            print(f"    {m.base_model_id}: {n}/1000")

    print(f"\n{'=' * 70}")
    print(
        f"Summary: {len(new_models)} new, {len(in_progress)} in progress, "
        f"{len(complete)} complete, {len(removed)} removed"
    )

    # Save state for next run
    save_state(free_models)

    # Auto-run if requested
    if not args.run:
        if new_models or in_progress:
            print("\nRun with --run to start/resume evaluation.")
        return

    targets = new_models + [m for m, _ in in_progress]
    if not targets:
        print("\nNothing to evaluate — all models complete.")
        return

    print(f"\nWill evaluate {len(targets)} model(s):")
    for m in targets:
        n = checkpoint_line_count(m.base_model_id)
        remaining = 1000 - n
        label = f"resume from {n}" if n > 0 else "new"
        print(f"  {m.base_model_id} ({label}, ~{remaining} images remaining)")

    # Load shared resources once
    download_test_data()
    ground_truth = load_ground_truth()
    gt_lookup = {gt.res_file: gt for gt in ground_truth}
    system_prompt = build_system_prompt()
    print(f"Loaded {len(ground_truth)} ground truth entries")

    all_metrics: dict[str, dict[str, Any]] = {}
    for m in targets:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {m.base_model_id} ({m.name})")
        print(f"{'=' * 70}")

        metrics = run_eval_for_model(
            base_model_id=m.base_model_id,
            ground_truth=ground_truth,
            api_key=api_key,
            system_prompt=system_prompt,
            gt_lookup=gt_lookup,
            limit=args.limit,
        )
        all_metrics[m.base_model_id] = metrics

    # Print summary
    if all_metrics:
        print(f"\n{'=' * 70}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 70}")
        print(f"{'Model':<45s} {'wSRCC':>7s} {'SRCC_O':>7s} {'n':>5s}")
        print("-" * 70)
        for model_id, m in sorted(
            all_metrics.items(),
            key=lambda x: x[1].get("wsrcc", -1),
            reverse=True,
        ):
            print(
                f"{model_id:<45s} "
                f"{m.get('wsrcc', 0):>7.4f} "
                f"{m.get('overall_srcc', 0):>7.4f} "
                f"{m.get('num_samples', 0):>5d}"
            )


if __name__ == "__main__":
    main()
