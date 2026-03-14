"""Statistical analysis for full-scale prompt arm validation.

Implements:
  1. Sub-sampling power curves (wSRCC SD vs sample size)
  2. Paired bootstrap CIs for deltas (arm vs baseline)
  3. Holm-Bonferroni correction for multiple comparisons
  4. Arm ranking stability analysis
  5. Per-quality-bucket analysis

Usage:
    cd DeQA-Score
    PYTHONPATH=./:$PYTHONPATH .venv/bin/python \
        ../results/vlm_teacher_eval/full_eval/analyze_prompt_power.py \
        --model qwen/qwen3.5-flash-02-23

    # Run all analyses:
    ... analyze_prompt_power.py --model qwen/qwen3.5-flash-02-23 --all

    # Just power curves:
    ... analyze_prompt_power.py --model qwen/qwen3.5-flash-02-23 --power

    # Just paired deltas:
    ... analyze_prompt_power.py --model qwen/qwen3.5-flash-02-23 --deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
TEST_CSV = DATA_DIR / "test.csv"
CHECKPOINT_DIR = EVAL_DIR / "checkpoints"
RESULTS_DIR = EVAL_DIR / "results"

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
SUBSAMPLE_DRAWS = 500
SUBSAMPLE_SIZES = [25, 50, 100, 150, 200, 300, 500, 750, 1000]

DIMS = ["overall", "sharpness", "color_fidelity"]
WSRCC_WEIGHTS = np.array([0.5, 0.25, 0.25])

# Quality buckets for stratified analysis
QUALITY_BUCKETS = [
    ("bad", 1.0, 1.8),
    ("poor", 1.8, 2.6),
    ("fair", 2.6, 3.4),
    ("good", 3.4, 4.0),
    ("excellent", 4.0, 5.01),
]


# --- Data loading ---


def load_ground_truth() -> dict[str, dict[str, float]]:
    """Load ground truth from test.csv."""
    gt: dict[str, dict[str, float]] = {}
    with TEST_CSV.open() as f:
        for row in csv.DictReader(f):
            gt[row["res"]] = {
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color_fidelity": float(row["color_fidelity"]),
            }
    return gt


def load_arm_checkpoint(model_id: str, arm_suffix: str) -> dict[str, dict[str, Any]]:
    """Load a checkpoint file, keeping valid records only."""
    safe_name = model_id.replace("/", "__")
    if arm_suffix == "arm1_baseline":
        cp_path = CHECKPOINT_DIR / f"{safe_name}.jsonl"
    else:
        cp_path = CHECKPOINT_DIR / f"{safe_name}__{arm_suffix}.jsonl"

    # Fallback: check legacy naming (e.g., __no_resize for arm7_no_resize)
    if not cp_path.exists():
        legacy_map = {"arm7_no_resize": "no_resize"}
        legacy_suffix = legacy_map.get(arm_suffix)
        if legacy_suffix:
            cp_path = CHECKPOINT_DIR / f"{safe_name}__{legacy_suffix}.jsonl"

    results: dict[str, dict[str, Any]] = {}
    if not cp_path.exists():
        return results
    for line in cp_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            img = item.get("image", "")
            if img and not item.get("error") and item.get("overall") is not None:
                results[img] = item
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_all_arms(model_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Load all available arm checkpoints for a model.

    Returns:
        Dict mapping arm_suffix -> {image -> record}.
    """
    arm_suffixes = [
        "arm1_baseline",
        "arm2_separate",
        "arm3_hybrid",
        "arm4_few_shot",
        "arm5_multi_sample",
        "arm6_res2048",
        "arm7_no_resize",
    ]
    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for suffix in arm_suffixes:
        data = load_arm_checkpoint(model_id, suffix)
        if data:
            arms[suffix] = data
            print(f"  Loaded {suffix}: {len(data)} valid records")
    return arms


# --- Core metric computation ---


def compute_wsrcc(
    pred_records: list[dict[str, Any]],
    gt: dict[str, dict[str, float]],
    image_indices: np.ndarray | None = None,
) -> float:
    """Compute wSRCC on a set of paired predictions.

    Args:
        pred_records: List of prediction dicts (must have 'image', 'overall', etc.)
        gt: Ground truth lookup.
        image_indices: If provided, select these indices from pred_records.

    Returns:
        Weighted SRCC value.
    """
    if image_indices is not None:
        records = [pred_records[i] for i in image_indices]
    else:
        records = pred_records

    srcc_values = []
    for dim in DIMS:
        pred = np.array([r[dim] for r in records])
        true = np.array([gt[r["image"]][dim] for r in records])
        srcc = float(stats.spearmanr(pred, true).statistic)
        srcc_values.append(srcc)

    return float(np.average(srcc_values, weights=WSRCC_WEIGHTS))


# --- Analysis 1: Sub-sampling power curves ---


def analyze_power_curves(
    arms: dict[str, dict[str, dict[str, Any]]],
    gt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute wSRCC variability and arm ranking stability at different sample sizes.

    For each sample size n:
    - Draw SUBSAMPLE_DRAWS stratified sub-samples
    - Compute wSRCC for each arm on each sub-sample
    - Record: mean, SD, and which arm wins

    Returns:
        Dict with power curve data.
    """
    print("\n--- Sub-sampling Power Analysis ---")

    # Get the common set of images across arms with full data (n>=100)
    full_arms = {k: v for k, v in arms.items() if len(v) >= 100}
    if len(full_arms) < 2:
        print("  Need at least 2 arms with >=100 images for power analysis")
        return {"error": "insufficient data"}
    common_images = set.intersection(*(set(arm.keys()) for arm in full_arms.values()))
    arms = full_arms
    print(f"  Common images across {len(arms)} arms with n>=100: {len(common_images)}")

    # Build image list with bucket assignments for stratified sampling
    image_list: list[str] = sorted(common_images)
    bucket_assignments: dict[str, int] = {}
    for img in image_list:
        mos = gt[img]["overall"]
        for bid, (_, lo, hi) in enumerate(QUALITY_BUCKETS):
            if lo <= mos < hi:
                bucket_assignments[img] = bid
                break

    # Group images by bucket
    bucket_images: dict[int, list[int]] = {i: [] for i in range(len(QUALITY_BUCKETS))}
    for idx, img in enumerate(image_list):
        bid = bucket_assignments.get(img, 2)  # default to "fair"
        bucket_images[bid].append(idx)

    bucket_sizes = {QUALITY_BUCKETS[k][0]: len(v) for k, v in bucket_images.items()}
    print(f"  Bucket sizes: {bucket_sizes}")

    # Build prediction arrays for each arm (ordered by image_list)
    arm_records: dict[str, list[dict[str, Any]]] = {}
    for arm_name, arm_data in arms.items():
        arm_records[arm_name] = [arm_data[img] for img in image_list]

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    results: dict[str, Any] = {"sample_sizes": SUBSAMPLE_SIZES, "arms": list(arms.keys())}

    for n in SUBSAMPLE_SIZES:
        if n > len(image_list):
            continue

        print(f"  n={n}: ", end="", flush=True)

        arm_wsrccs: dict[str, list[float]] = {arm: [] for arm in arms}
        winner_counts: dict[str, int] = {arm: 0 for arm in arms}

        for _ in range(SUBSAMPLE_DRAWS):
            # Stratified sampling: proportional allocation
            selected_indices: list[int] = []
            for bid, indices in bucket_images.items():
                n_bucket = max(1, round(len(indices) / len(image_list) * n))
                n_bucket = min(n_bucket, len(indices))
                chosen = rng.choice(indices, size=n_bucket, replace=False)
                selected_indices.extend(chosen)

            # Trim or pad to exactly n
            if len(selected_indices) > n:
                selected_indices = list(rng.choice(selected_indices, size=n, replace=False))
            idx_arr = np.array(selected_indices)

            # Compute wSRCC for each arm
            draw_wsrccs: dict[str, float] = {}
            for arm_name, records in arm_records.items():
                try:
                    w = compute_wsrcc(records, gt, idx_arr)
                    if not np.isnan(w):
                        arm_wsrccs[arm_name].append(w)
                        draw_wsrccs[arm_name] = w
                except (ValueError, FloatingPointError):
                    pass

            # Who wins this draw?
            if draw_wsrccs:
                best = max(draw_wsrccs, key=draw_wsrccs.get)  # type: ignore[arg-type]
                winner_counts[best] = winner_counts.get(best, 0) + 1

        # Summarize
        n_result: dict[str, Any] = {"n": n}
        for arm_name in arms:
            vals = arm_wsrccs[arm_name]
            if vals:
                n_result[arm_name] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "sd": round(float(np.std(vals)), 4),
                    "ci_95_width": round(float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5)), 4),
                }

        n_result["ranking_stability"] = {
            arm: round(count / SUBSAMPLE_DRAWS, 4)
            for arm, count in winner_counts.items()
        }

        results[f"n_{n}"] = n_result
        print(f"done (baseline SD={n_result.get('arm1_baseline', {}).get('sd', '?')})")

    return results


# --- Analysis 2: Paired bootstrap deltas ---


def analyze_paired_deltas(
    arms: dict[str, dict[str, dict[str, Any]]],
    gt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute paired bootstrap CIs for wSRCC deltas (each arm vs baseline).

    Uses image-as-unit resampling: for each bootstrap replicate, resample
    image indices and compute delta = wSRCC_arm - wSRCC_baseline on the
    SAME resampled set.
    """
    print("\n--- Paired Bootstrap Delta CIs ---")

    baseline_key = "arm1_baseline"
    if baseline_key not in arms:
        print("  ERROR: No baseline checkpoint found")
        return {}

    results: dict[str, Any] = {}

    for arm_name, arm_data in arms.items():
        if arm_name == baseline_key:
            continue

        # Find common images between this arm and baseline
        common = sorted(set(arms[baseline_key].keys()) & set(arm_data.keys()))
        n = len(common)
        if n < 30:
            print(f"  {arm_name}: too few common images ({n})")
            continue

        # Build ordered arrays
        baseline_records = [arms[baseline_key][img] for img in common]
        arm_records = [arm_data[img] for img in common]

        # Point estimates
        baseline_wsrcc = compute_wsrcc(baseline_records, gt)
        arm_wsrcc = compute_wsrcc(arm_records, gt)
        point_delta = arm_wsrcc - baseline_wsrcc

        # Bootstrap
        rng = np.random.RandomState(BOOTSTRAP_SEED)
        boot_deltas: list[float] = []

        for _ in range(BOOTSTRAP_N):
            idx = rng.randint(0, n, size=n)
            try:
                b_wsrcc = compute_wsrcc(baseline_records, gt, idx)
                a_wsrcc = compute_wsrcc(arm_records, gt, idx)
                delta = a_wsrcc - b_wsrcc
                if not np.isnan(delta):
                    boot_deltas.append(delta)
            except (ValueError, FloatingPointError):
                continue

        if len(boot_deltas) < 100:
            print(f"  {arm_name}: insufficient bootstrap samples ({len(boot_deltas)})")
            continue

        ci_lo = float(np.percentile(boot_deltas, 2.5))
        ci_hi = float(np.percentile(boot_deltas, 97.5))
        p_positive = float(np.mean([d > 0 for d in boot_deltas]))

        results[arm_name] = {
            "n_common": n,
            "baseline_wsrcc": round(baseline_wsrcc, 4),
            "arm_wsrcc": round(arm_wsrcc, 4),
            "delta": round(point_delta, 4),
            "delta_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "p_positive": round(p_positive, 4),
            "boot_sd": round(float(np.std(boot_deltas)), 4),
        }

        sig = "*" if ci_lo > 0 or ci_hi < 0 else ""
        print(
            f"  {arm_name}: delta={point_delta:+.4f} "
            f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
            f"P(delta>0)={p_positive:.3f} {sig}"
        )

    # Apply Holm-Bonferroni correction
    if results:
        results = apply_holm_correction(results)

    return results


def apply_holm_correction(
    delta_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply Holm-Bonferroni correction to paired delta results.

    Uses 1 - P(delta>0) or P(delta>0) (whichever is smaller) as the
    two-sided p-value proxy, then applies Holm step-down procedure.
    """
    print("\n  Holm-Bonferroni correction:")

    # Compute two-sided p-values from bootstrap
    arm_pvals: list[tuple[str, float]] = []
    for arm_name, data in delta_results.items():
        p_pos = data["p_positive"]
        p_twosided = 2 * min(p_pos, 1 - p_pos)  # two-sided
        p_twosided = min(p_twosided, 1.0)
        arm_pvals.append((arm_name, p_twosided))
        data["p_twosided"] = round(p_twosided, 4)

    # Sort by p-value (ascending)
    arm_pvals.sort(key=lambda x: x[1])
    m = len(arm_pvals)

    for rank, (arm_name, p_raw) in enumerate(arm_pvals):
        p_adjusted = min(p_raw * (m - rank), 1.0)
        delta_results[arm_name]["p_holm"] = round(p_adjusted, 4)
        sig = "***" if p_adjusted < 0.001 else "**" if p_adjusted < 0.01 else "*" if p_adjusted < 0.05 else ""
        print(f"    {arm_name}: p_raw={p_raw:.4f} p_holm={p_adjusted:.4f} {sig}")

    return delta_results


# --- Analysis 3: Per-quality-bucket breakdown ---


def analyze_per_bucket(
    arms: dict[str, dict[str, dict[str, Any]]],
    gt: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Compute per-quality-bucket SRCC for each arm."""
    print("\n--- Per-Quality-Bucket Analysis ---")

    results: dict[str, Any] = {}

    for arm_name, arm_data in arms.items():
        arm_results: dict[str, Any] = {}

        for bucket_name, lo, hi in QUALITY_BUCKETS:
            # Get images in this bucket that have predictions
            bucket_imgs = [
                img for img, g in gt.items()
                if lo <= g["overall"] < hi and img in arm_data
            ]

            if len(bucket_imgs) < 10:
                arm_results[bucket_name] = {"n": len(bucket_imgs), "note": "too few"}
                continue

            records = [arm_data[img] for img in bucket_imgs]
            bucket_gt = {img: gt[img] for img in bucket_imgs}

            # Compute per-dimension SRCC
            dim_srcc: dict[str, float] = {}
            for dim in DIMS:
                pred = np.array([r[dim] for r in records])
                true = np.array([bucket_gt[r["image"]][dim] for r in records])
                try:
                    srcc = float(stats.spearmanr(pred, true).statistic)
                    dim_srcc[dim] = round(srcc, 4)
                except (ValueError, FloatingPointError):
                    dim_srcc[dim] = float("nan")

            arm_results[bucket_name] = {
                "n": len(bucket_imgs),
                "overall_srcc": dim_srcc.get("overall", float("nan")),
                "sharpness_srcc": dim_srcc.get("sharpness", float("nan")),
                "color_srcc": dim_srcc.get("color_fidelity", float("nan")),
            }

        results[arm_name] = arm_results

    # Print summary
    print(f"\n  {'Arm':<20s}", end="")
    for bname, _, _ in QUALITY_BUCKETS:
        print(f" {bname:>10s}", end="")
    print()
    print("  " + "-" * 75)

    for arm_name, arm_results in results.items():
        print(f"  {arm_name:<20s}", end="")
        for bname, _, _ in QUALITY_BUCKETS:
            br = arm_results.get(bname, {})
            srcc = br.get("overall_srcc", float("nan"))
            n = br.get("n", 0)
            if isinstance(srcc, float) and not np.isnan(srcc):
                print(f" {srcc:>7.3f}({n:>2d})", end="")
            else:
                print(f"       -({n:>2d})", end="")
        print()

    return results


# --- Analysis 4: Failure rate analysis ---


def analyze_failure_rates(
    model_id: str,
) -> dict[str, Any]:
    """Analyze parse/API failure rates per arm."""
    print("\n--- Failure Rate Analysis ---")

    arm_suffixes = [
        ("arm1_baseline", ""),
        ("arm2_separate", "arm2_separate"),
        ("arm3_hybrid", "arm3_hybrid"),
        ("arm4_few_shot", "arm4_few_shot"),
        ("arm5_multi_sample", "arm5_multi_sample"),
        ("arm6_res2048", "arm6_res2048"),
        ("arm7_no_resize", "arm7_no_resize"),
    ]

    safe_name = model_id.replace("/", "__")
    results: dict[str, Any] = {}

    for arm_name, suffix in arm_suffixes:
        if suffix:
            cp_path = CHECKPOINT_DIR / f"{safe_name}__{suffix}.jsonl"
        else:
            cp_path = CHECKPOINT_DIR / f"{safe_name}.jsonl"

        if not cp_path.exists():
            continue

        total = 0
        errors = 0
        null_scores = 0
        for line in cp_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                total += 1
                if item.get("error"):
                    errors += 1
                elif item.get("overall") is None:
                    null_scores += 1
            except json.JSONDecodeError:
                total += 1
                errors += 1

        if total > 0:
            results[arm_name] = {
                "total": total,
                "api_errors": errors,
                "parse_errors": null_scores,
                "success_rate": round((total - errors - null_scores) / total, 4),
            }
            print(
                f"  {arm_name:<20s}: {total:>5d} total, "
                f"{errors:>3d} API errors, {null_scores:>3d} parse errors, "
                f"success={results[arm_name]['success_rate']:.1%}"
            )

    return results


# --- Main ---


def main() -> None:
    """Run statistical analyses on prompt arm results."""
    parser = argparse.ArgumentParser(
        description="Statistical analysis for prompt arm validation"
    )
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--power", action="store_true", help="Sub-sampling power curves")
    parser.add_argument("--deltas", action="store_true", help="Paired bootstrap deltas")
    parser.add_argument("--buckets", action="store_true", help="Per-quality-bucket analysis")
    parser.add_argument("--failures", action="store_true", help="Failure rate analysis")
    args = parser.parse_args()

    if args.all:
        args.power = args.deltas = args.buckets = args.failures = True
    if not any([args.power, args.deltas, args.buckets, args.failures]):
        parser.error("Specify --all or at least one of --power, --deltas, --buckets, --failures")

    model_id: str = args.model
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} images")

    arms = load_all_arms(model_id)
    if not arms:
        print("No arm data found. Run run_full_prompt_arms.py first.")
        sys.exit(1)

    all_results: dict[str, Any] = {"model": model_id, "n_arms": len(arms)}

    if args.power:
        all_results["power_curves"] = analyze_power_curves(arms, gt)

    if args.deltas:
        all_results["paired_deltas"] = analyze_paired_deltas(arms, gt)

    if args.buckets:
        all_results["per_bucket"] = analyze_per_bucket(arms, gt)

    if args.failures:
        all_results["failure_rates"] = analyze_failure_rates(model_id)

    # Save
    out_path = RESULTS_DIR / f"prompt_power_{model_id.replace('/', '__')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
