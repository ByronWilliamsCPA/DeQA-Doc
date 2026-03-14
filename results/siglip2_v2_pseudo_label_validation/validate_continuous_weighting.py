#!/usr/bin/env python3
"""Step 2.1c: Compare continuous weighting schemes vs discrete tiers.

Tests whether sigma_sq-based continuous weights improve pseudo-label
utility compared to discrete tier assignment. Evaluates by simulating
a weighted training scenario: if we weight each sample by its confidence,
does the effective dataset quality improve?

Key metric: "effective MAE" = weighted mean of |pseudo - GT|, where
lower = better pseudo-labels for training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent / "DeQA-Score"
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ("overall", "sharpness", "color")


def load_all():
    """Load predictions, GT, signals."""
    from validate_pseudo_labels import (
        load_siglip2_val_predictions,
        load_val_embeddings,
        load_val_gt,
    )
    from validate_tiering import collect_signals

    siglip2_outputs = load_siglip2_val_predictions()
    gt = load_val_gt()
    embeddings, _ = load_val_embeddings()
    records = collect_signals(siglip2_outputs, embeddings)
    return records, gt


# --- Weighting schemes ---

def weight_uniform(records: list[dict]) -> np.ndarray:
    """Baseline: all samples weight=1."""
    return np.ones(len(records))


def weight_discrete_tiers(records: list[dict], dim: str) -> np.ndarray:
    """Current fusion-style discrete tiers based on percentiles."""
    sigma_sqs = np.array([r["sigma_sq"] for r in records])
    entropies = np.array([r["entropy"] for r in records])
    d_ms = np.array([r["d_m"] for r in records])

    def rank_norm(arr):
        ranks = stats.rankdata(arr)
        return (ranks - 1) / (len(ranks) - 1)

    combined = (rank_norm(d_ms) + rank_norm(sigma_sqs) + rank_norm(entropies)) / 3.0

    weights = np.ones(len(records))
    p50 = np.percentile(combined, 50)
    p80 = np.percentile(combined, 80)
    p95 = np.percentile(combined, 95)

    weights[combined > p50] = 0.6   # low_weight
    weights[combined > p80] = 0.3   # tier2
    weights[combined > p95] = 0.0   # hard_reject
    return weights


def weight_sigma_linear(records: list[dict], clip_hi: float = None) -> np.ndarray:
    """Linear decay: w = 1 - (sigma_sq - min) / (max - min)."""
    sigma_sqs = np.array([r["sigma_sq"] for r in records])
    lo = sigma_sqs.min()
    hi = clip_hi or sigma_sqs.max()
    normalized = np.clip((sigma_sqs - lo) / (hi - lo), 0, 1)
    return 1.0 - 0.7 * normalized  # range [0.3, 1.0]


def weight_sigma_exp(records: list[dict], tau: float = None) -> np.ndarray:
    """Exponential decay: w = exp(-sigma_sq / tau)."""
    sigma_sqs = np.array([r["sigma_sq"] for r in records])
    if tau is None:
        tau = np.median(sigma_sqs)
    raw = np.exp(-sigma_sqs / tau)
    # Rescale to [0.3, 1.0]
    lo, hi = raw.min(), raw.max()
    if hi - lo < 1e-8:
        return np.ones(len(records))
    return 0.3 + 0.7 * (raw - lo) / (hi - lo)


def weight_sigma_entropy_combined(records: list[dict]) -> np.ndarray:
    """Combined sigma_sq + entropy via rank-average, linear decay."""
    sigma_sqs = np.array([r["sigma_sq"] for r in records])
    entropies = np.array([r["entropy"] for r in records])

    def rank_norm(arr):
        ranks = stats.rankdata(arr)
        return (ranks - 1) / (len(ranks) - 1)

    combined = (rank_norm(sigma_sqs) + rank_norm(entropies)) / 2.0
    return 1.0 - 0.7 * combined  # [0.3, 1.0]


def weight_sigma_quantile(records: list[dict]) -> np.ndarray:
    """Quantile-based: map sigma_sq rank to weight via smooth step."""
    sigma_sqs = np.array([r["sigma_sq"] for r in records])
    ranks = stats.rankdata(sigma_sqs)
    quantiles = (ranks - 1) / (len(ranks) - 1)  # [0, 1]
    # Smooth step: steep decay for top 20%
    weights = np.where(
        quantiles < 0.5,
        1.0,
        np.where(
            quantiles < 0.8,
            1.0 - 0.5 * (quantiles - 0.5) / 0.3,  # 1.0 -> 0.5
            0.5 - 0.2 * (quantiles - 0.8) / 0.2,   # 0.5 -> 0.3
        ),
    )
    return weights


def evaluate_scheme(
    records: list[dict],
    gt: dict[str, dict[str, float]],
    weights: np.ndarray,
) -> dict:
    """Evaluate a weighting scheme: weighted MAE, effective N, etc."""
    errors = np.array([abs(r["mos"] - gt[r["image_id"]][r["dimension"]]) for r in records])

    # Weighted MAE (what training loss effectively sees)
    weighted_mae = np.average(errors, weights=weights)

    # Unweighted MAE of accepted samples (weight > 0.5)
    high_conf_mask = weights >= 0.5
    high_conf_mae = errors[high_conf_mask].mean() if high_conf_mask.any() else float("nan")
    high_conf_n = int(high_conf_mask.sum())

    # Effective sample size: (sum w)^2 / sum(w^2)
    eff_n = (weights.sum() ** 2) / (weights ** 2).sum()

    # Weighted SRCC (rank correlation on weighted subset)
    # Use samples with weight > 0.3 for correlation
    active_mask = weights > 0.3
    if active_mask.sum() >= 20:
        pred = np.array([r["mos"] for r, m in zip(records, active_mask) if m])
        gt_arr = np.array([gt[r["image_id"]][r["dimension"]] for r, m in zip(records, active_mask) if m])
        srcc, _ = stats.spearmanr(pred, gt_arr)
    else:
        srcc = float("nan")

    return {
        "weighted_mae": float(weighted_mae),
        "high_conf_mae": float(high_conf_mae),
        "high_conf_n": high_conf_n,
        "effective_n": float(eff_n),
        "mean_weight": float(weights.mean()),
        "min_weight": float(weights.min()),
        "active_srcc": float(srcc),
        "pct_full_weight": float((weights >= 0.95).sum() / len(weights) * 100),
        "pct_downweighted": float(((weights > 0.3) & (weights < 0.95)).sum() / len(weights) * 100),
        "pct_rejected": float((weights <= 0.3).sum() / len(weights) * 100),
    }


def main():
    print("Loading data...")
    records, gt = load_all()

    # Filter to matched records
    records = [r for r in records if r["image_id"] in gt]

    schemes = {
        "uniform (baseline)": weight_uniform,
        "discrete tiers": lambda recs: weight_discrete_tiers(recs, "all"),
        "sigma_sq linear": weight_sigma_linear,
        "sigma_sq exponential": weight_sigma_exp,
        "sigma_sq + entropy": weight_sigma_entropy_combined,
        "sigma_sq quantile-step": weight_sigma_quantile,
    }

    print("\n" + "=" * 90)
    print("WEIGHTING SCHEME COMPARISON (all dimensions pooled)")
    print("=" * 90)

    print(f"\n{'Scheme':<25s} {'Wtd MAE':>8s} {'Hi-conf MAE':>12s} {'Hi-conf N':>10s}"
          f" {'Eff N':>8s} {'SRCC':>8s} {'Full%':>7s} {'Down%':>7s} {'Rej%':>6s}")
    print(f"{'-'*25} {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")

    all_results = {}
    for name, scheme_fn in schemes.items():
        weights = scheme_fn(records)
        result = evaluate_scheme(records, gt, weights)
        all_results[name] = result

        print(f"{name:<25s} {result['weighted_mae']:8.4f} {result['high_conf_mae']:12.4f}"
              f" {result['high_conf_n']:10d} {result['effective_n']:8.1f}"
              f" {result['active_srcc']:8.4f} {result['pct_full_weight']:6.1f}%"
              f" {result['pct_downweighted']:6.1f}% {result['pct_rejected']:5.1f}%")

    # Per-dimension breakdown for top schemes
    print("\n" + "=" * 90)
    print("PER-DIMENSION BREAKDOWN (top schemes)")
    print("=" * 90)

    top_schemes = ["uniform (baseline)", "sigma_sq exponential", "sigma_sq + entropy"]

    for dim in DIMENSIONS:
        dim_recs = [r for r in records if r["dimension"] == dim]
        print(f"\n--- {dim.upper()} (n={len(dim_recs)}) ---")
        print(f"  {'Scheme':<25s} {'Wtd MAE':>8s} {'Hi-conf MAE':>12s}"
              f" {'Hi-conf N':>10s} {'Eff N':>8s}")
        print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*10} {'-'*8}")

        for name in top_schemes:
            weights = schemes[name](dim_recs)
            result = evaluate_scheme(dim_recs, gt, weights)
            print(f"  {name:<25s} {result['weighted_mae']:8.4f}"
                  f" {result['high_conf_mae']:12.4f} {result['high_conf_n']:10d}"
                  f" {result['effective_n']:8.1f}")

    # Pareto analysis: MAE reduction vs effective N tradeoff
    print("\n" + "=" * 90)
    print("PARETO ANALYSIS: MAE reduction vs data efficiency")
    print("=" * 90)
    baseline_mae = all_results["uniform (baseline)"]["weighted_mae"]
    baseline_n = all_results["uniform (baseline)"]["effective_n"]

    print(f"\n{'Scheme':<25s} {'MAE reduction':>14s} {'N retention':>12s} {'MAE/N ratio':>12s}")
    print(f"{'-'*25} {'-'*14} {'-'*12} {'-'*12}")
    for name, result in all_results.items():
        mae_reduction = (1 - result["weighted_mae"] / baseline_mae) * 100
        n_retention = result["effective_n"] / baseline_n * 100
        ratio = mae_reduction / (100 - n_retention + 1e-8) if n_retention < 100 else 0
        print(f"{name:<25s} {mae_reduction:+13.2f}% {n_retention:11.1f}%"
              f" {ratio:12.3f}")

    # Save results
    output_path = Path(__file__).resolve().parent / "weighting_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
