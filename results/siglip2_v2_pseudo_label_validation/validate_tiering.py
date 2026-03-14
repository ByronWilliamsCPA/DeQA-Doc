#!/usr/bin/env python3
"""Step 2.1b: Evaluate tiering system by tightening auto-accept thresholds.

Instead of using default thresholds (which auto-accept everything on ID data),
we sweep tighter thresholds to see how well the uncertainty signals stratify
pseudo-label quality. If tiering works, lower-tier samples should have higher
MAE against ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "DeQA-Score"
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ("overall", "sharpness", "color")


def load_data():
    """Load all inputs and return (siglip2_outputs, gt, embeddings)."""
    from validate_pseudo_labels import (
        load_siglip2_val_predictions,
        load_val_embeddings,
        load_val_gt,
    )

    siglip2_outputs = load_siglip2_val_predictions()
    gt = load_val_gt()
    embeddings, _ = load_val_embeddings()
    return siglip2_outputs, gt, embeddings


def collect_signals(siglip2_outputs, embeddings):
    """Run pipeline and collect per-sample signals + pseudo-label MOS."""
    from src.uncertainty.cross_validator import CrossValidator
    from src.uncertainty.fusion import UncertaintyFusion
    from src.uncertainty.gaussian_to_discrete import (
        level_probs_to_mos,
        siglip2_output_to_level_probs,
    )
    from src.uncertainty.ood_wrapper import OODDetectorWrapper

    ood_path = RESULTS_ROOT / "siglip2_diqa5000" / "ood_detector_v2.npz"
    ood_detector = OODDetectorWrapper.from_npz(str(ood_path))
    CrossValidator({})
    fusion = UncertaintyFusion()

    records = []
    for i, output in enumerate(siglip2_outputs):
        image_id = output["image_id"]
        embedding = embeddings[i]

        # OOD score (shared across dimensions)
        ood_result = ood_detector.score(embedding)
        d_m = ood_result.mahalanobis_distance

        for dim in DIMENSIONS:
            mu = output[f"{dim}_mu"]
            sigma_sq = output[f"{dim}_sigma_sq"]

            # Discretize and compute entropy
            level_probs = siglip2_output_to_level_probs(mu, sigma_sq)
            mos = level_probs_to_mos(level_probs)

            from src.uncertainty.discrete_metrics import discrete_entropy

            entropy = discrete_entropy(level_probs)

            records.append({
                "image_id": image_id,
                "dimension": dim,
                "mu": mu,
                "sigma_sq": sigma_sq,
                "mos": mos,
                "d_m": d_m,
                "entropy": entropy,
                "level_probs": level_probs,
            })

    return records


def analyze_signal_distributions(records, gt):
    """Print signal distributions and correlation with GT error."""
    print("\n" + "=" * 70)
    print("SIGNAL DISTRIBUTIONS")
    print("=" * 70)

    for dim in DIMENSIONS:
        dim_recs = [r for r in records if r["dimension"] == dim]
        errors = []
        d_ms = []
        sigma_sqs = []
        entropies = []

        for r in dim_recs:
            if r["image_id"] not in gt:
                continue
            gt_mos = gt[r["image_id"]][dim]
            errors.append(abs(r["mos"] - gt_mos))
            d_ms.append(r["d_m"])
            sigma_sqs.append(r["sigma_sq"])
            entropies.append(r["entropy"])

        errors = np.array(errors)
        d_ms = np.array(d_ms)
        sigma_sqs = np.array(sigma_sqs)
        entropies = np.array(entropies)

        print(f"\n--- {dim.upper()} (n={len(errors)}) ---")
        for name, vals in [
            ("d_M (Mahalanobis)", d_ms),
            ("sigma_sq", sigma_sqs),
            ("entropy", entropies),
            ("abs_error", errors),
        ]:
            print(f"  {name:25s}: min={vals.min():.4f}  p25={np.percentile(vals, 25):.4f}"
                  f"  p50={np.percentile(vals, 50):.4f}  p75={np.percentile(vals, 75):.4f}"
                  f"  p90={np.percentile(vals, 90):.4f}  max={vals.max():.4f}")

        # Correlation of each signal with absolute error
        print("\n  Signal-error correlations (higher = signal predicts error):")
        for name, vals in [
            ("d_M", d_ms),
            ("sigma_sq", sigma_sqs),
            ("entropy", entropies),
        ]:
            srcc, p = stats.spearmanr(vals, errors)
            print(f"    {name:15s} vs |error|: SRCC={srcc:+.4f}  (p={p:.2e})")


def simulate_tiering(records, gt):
    """Simulate tiering at various percentile-based thresholds."""
    print("\n" + "=" * 70)
    print("TIERING SIMULATION")
    print("Assigning tiers by percentile cutoffs on combined uncertainty score")
    print("=" * 70)

    for dim in DIMENSIONS:
        dim_recs = [r for r in records if r["dimension"] == dim and r["image_id"] in gt]

        # Compute a combined uncertainty score (rank-average of signals)
        d_ms = np.array([r["d_m"] for r in dim_recs])
        sigma_sqs = np.array([r["sigma_sq"] for r in dim_recs])
        entropies = np.array([r["entropy"] for r in dim_recs])
        errors = np.array([abs(r["mos"] - gt[r["image_id"]][dim]) for r in dim_recs])

        # Rank-normalize each signal to [0, 1]
        def rank_normalize(arr):
            ranks = stats.rankdata(arr)
            return (ranks - 1) / (len(ranks) - 1)

        combined = (
            rank_normalize(d_ms)
            + rank_normalize(sigma_sqs)
            + rank_normalize(entropies)
        ) / 3.0

        # Tier by percentile: auto_accept (bottom 50%), low_weight (50-80%),
        # tier2 (80-95%), hard_reject (top 5%)
        thresholds = [
            ("auto_accept", 0.0, 50.0),
            ("low_weight", 50.0, 80.0),
            ("tier2_trigger", 80.0, 95.0),
            ("hard_reject", 95.0, 100.0),
        ]

        print(f"\n--- {dim.upper()} ---")
        print(f"  {'Tier':<20s} {'Count':>6s} {'MAE':>8s} {'RMSE':>8s}"
              f" {'SRCC':>8s} {'Median err':>10s} {'p90 err':>10s}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")

        for tier_name, lo_pct, hi_pct in thresholds:
            lo_val = np.percentile(combined, lo_pct)
            hi_val = np.percentile(combined, hi_pct)
            if lo_pct == 0.0:
                mask = combined <= hi_val
            elif hi_pct == 100.0:
                mask = combined > lo_val
            else:
                mask = (combined > lo_val) & (combined <= hi_val)

            tier_errors = errors[mask]
            if len(tier_errors) == 0:
                continue

            mae = tier_errors.mean()
            rmse = np.sqrt((tier_errors**2).mean())
            median_e = np.median(tier_errors)
            p90_e = np.percentile(tier_errors, 90)

            srcc_str = "n/a"
            if mask.sum() >= 10:
                pred = np.array([dim_recs[j]["mos"] for j in range(len(dim_recs)) if mask[j]])
                gt_arr = np.array([gt[dim_recs[j]["image_id"]][dim] for j in range(len(dim_recs)) if mask[j]])
                s, _ = stats.spearmanr(pred, gt_arr)
                srcc_str = f"{s:.4f}"

            print(f"  {tier_name:<20s} {mask.sum():6d} {mae:8.4f} {rmse:8.4f}"
                  f" {srcc_str:>8s} {median_e:10.4f} {p90_e:10.4f}")

    # Also try individual signal thresholds
    print("\n" + "=" * 70)
    print("INDIVIDUAL SIGNAL TIERING")
    print("Splitting at each signal's median: low-uncertainty vs high-uncertainty")
    print("=" * 70)

    for dim in DIMENSIONS:
        dim_recs = [r for r in records if r["dimension"] == dim and r["image_id"] in gt]
        errors = np.array([abs(r["mos"] - gt[r["image_id"]][dim]) for r in dim_recs])

        print(f"\n--- {dim.upper()} ---")
        print(f"  {'Signal':<20s} {'Low-unc MAE':>12s} {'High-unc MAE':>13s}"
              f" {'Ratio':>8s} {'p-value':>12s}")
        print(f"  {'-'*20} {'-'*12} {'-'*13} {'-'*8} {'-'*12}")

        for name, vals in [
            ("d_M", np.array([r["d_m"] for r in dim_recs])),
            ("sigma_sq", np.array([r["sigma_sq"] for r in dim_recs])),
            ("entropy", np.array([r["entropy"] for r in dim_recs])),
        ]:
            median_val = np.median(vals)
            lo_mask = vals <= median_val
            hi_mask = vals > median_val

            lo_mae = errors[lo_mask].mean()
            hi_mae = errors[hi_mask].mean()
            ratio = hi_mae / lo_mae if lo_mae > 0 else float("inf")

            # Mann-Whitney U test
            _, p = stats.mannwhitneyu(errors[lo_mask], errors[hi_mask], alternative="less")

            print(f"  {name:<20s} {lo_mae:12.4f} {hi_mae:13.4f}"
                  f" {ratio:8.2f}x {p:12.2e}")


if __name__ == "__main__":
    print("Loading data...")
    siglip2_outputs, gt, embeddings = load_data()

    print("Collecting per-sample signals...")
    records = collect_signals(siglip2_outputs, embeddings)

    analyze_signal_distributions(records, gt)
    simulate_tiering(records, gt)
