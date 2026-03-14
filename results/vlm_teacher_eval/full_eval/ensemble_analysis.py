"""Ensemble analysis of VLM teacher models for DIQA pseudo-labeling."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent
CHECKPOINTS = BASE / "checkpoints"
GT_PATH = BASE / "data" / "test.csv"
DIMS = ["overall", "sharpness", "color_fidelity"]


def load_gt():
    """Load ground truth from test.csv."""
    gt = {}
    with open(GT_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["res"]
            gt[img] = {d: float(row[d]) for d in DIMS}
    return gt


def load_checkpoint(path):
    """Load predictions from a JSONL checkpoint."""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error"):
                continue
            img = rec["image"]
            vals = {}
            valid = True
            for d in DIMS:
                v = rec.get(d)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    valid = False
                    break
                vals[d] = float(v)
            if valid:
                preds[img] = vals
    return preds


def compute_metrics(preds, gt, images):
    """Compute per-dimension SRCC and wSRCC on the given image set."""
    results = {}
    for d in DIMS:
        y_true = np.array([gt[img][d] for img in images])
        y_pred = np.array([preds[img][d] for img in images])
        srcc, _ = stats.spearmanr(y_pred, y_true)
        results[d] = srcc
    results["wSRCC"] = (
        0.5 * results["overall"]
        + 0.25 * results["sharpness"]
        + 0.25 * results["color_fidelity"]
    )
    results["n"] = len(images)
    return results


def ensemble_preds(model_preds_list, images):
    """Simple mean ensemble over models for the given images."""
    ens = {}
    for img in images:
        ens[img] = {}
        for d in DIMS:
            ens[img][d] = np.mean([mp[img][d] for mp in model_preds_list])
    return ens


def pairwise_residual_corr(model_preds, gt, images):
    """Compute pairwise Pearson correlation of residuals between models."""
    names = list(model_preds.keys())
    # Compute residuals per model per dimension
    residuals = {}
    for name in names:
        residuals[name] = {}
        for d in DIMS:
            residuals[name][d] = np.array(
                [model_preds[name][img][d] - gt[img][d] for img in images]
            )

    print("\n" + "=" * 80)
    print("PAIRWISE RESIDUAL CORRELATIONS (Pearson r, averaged across dimensions)")
    print("=" * 80)

    # Header
    col_w = 14
    header = " " * 30
    for n in names:
        short = n.split("/")[-1][:col_w]
        header += f"{short:>{col_w}}"
    print(header)

    for i, n1 in enumerate(names):
        short1 = n1[:28]
        row = f"{short1:<30}"
        for j, n2 in enumerate(names):
            if j < i:
                # Compute average Pearson r across dimensions
                corrs = []
                for d in DIMS:
                    r, _ = stats.pearsonr(residuals[n1][d], residuals[n2][d])
                    corrs.append(r)
                avg_r = np.mean(corrs)
                row += f"{avg_r:>{col_w}.3f}"
            elif j == i:
                row += f"{'1.000':>{col_w}}"
            else:
                row += f"{'':>{col_w}}"
        print(row)

    # Also print per-dimension detail
    for d in DIMS:
        print(f"\n  [{d}]")
        header = " " * 30
        for n in names:
            short = n.split("/")[-1][:col_w]
            header += f"{short:>{col_w}}"
        print(header)
        for i, n1 in enumerate(names):
            short1 = n1[:28]
            row = f"{short1:<30}"
            for j, n2 in enumerate(names):
                if j <= i:
                    r, _ = stats.pearsonr(residuals[n1][d], residuals[n2][d])
                    row += f"{r:>{col_w}.3f}"
                else:
                    row += f"{'':>{col_w}}"
            print(row)


def main():
    gt = load_gt()
    print(f"Ground truth: {len(gt)} images")

    # Define models
    models = {
        "Flash": CHECKPOINTS / "google__gemini-3-flash-preview.jsonl",
        "GPT-4.1": CHECKPOINTS / "openai__gpt-4.1.jsonl",
        "Qwen122B(arm7)": CHECKPOINTS / "qwen__qwen3.5-122b-a10b__arm7_no_resize.jsonl",
        "Qwen122B(base)": CHECKPOINTS / "qwen__qwen3.5-122b-a10b.jsonl",
        "Lite(arm10)": CHECKPOINTS / "google__gemini-3.1-flash-lite-preview__arm10_combined.jsonl",
        "Lite(base)": CHECKPOINTS / "google__gemini-3.1-flash-lite-preview.jsonl",
    }

    preds = {}
    for name, path in models.items():
        preds[name] = load_checkpoint(path)
        valid = len(set(preds[name].keys()) & set(gt.keys()))
        print(f"  {name}: {len(preds[name])} predictions, {valid} matched to GT")

    # Individual model metrics (on full GT intersection)
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("=" * 80)
    print(f"{'Model':<25} {'N':>5} {'overall':>10} {'sharpness':>10} {'color':>10} {'wSRCC':>10}")
    print("-" * 75)
    for name in models:
        common = sorted(set(preds[name].keys()) & set(gt.keys()))
        m = compute_metrics(preds[name], gt, common)
        print(
            f"{name:<25} {m['n']:>5} {m['overall']:>10.4f} {m['sharpness']:>10.4f} "
            f"{m['color_fidelity']:>10.4f} {m['wSRCC']:>10.4f}"
        )

    # Define ensemble combinations
    # Use best variants: Flash (baseline), GPT-4.1, Qwen122B(arm7), Lite(arm10)
    ensemble_models = {
        "Flash": preds["Flash"],
        "GPT-4.1": preds["GPT-4.1"],
        "Qwen122B": preds["Qwen122B(arm7)"],
        "Lite(comb)": preds["Lite(arm10)"],
    }

    ensembles = {
        "Flash + GPT + Qwen122B": ["Flash", "GPT-4.1", "Qwen122B"],
        "Flash + Lite + GPT + Qwen122B": ["Flash", "Lite(comb)", "GPT-4.1", "Qwen122B"],
        "Flash + Lite + Qwen122B": ["Flash", "Lite(comb)", "Qwen122B"],
        "Lite + GPT + Qwen122B": ["Lite(comb)", "GPT-4.1", "Qwen122B"],
        "Lite + Qwen122B": ["Lite(comb)", "Qwen122B"],
        "Flash + Qwen122B": ["Flash", "Qwen122B"],
        "Lite(comb) solo": ["Lite(comb)"],
        "Flash solo": ["Flash"],
        "GPT-4.1 solo": ["GPT-4.1"],
        "Qwen122B solo": ["Qwen122B"],
    }

    print("\n" + "=" * 80)
    print("ENSEMBLE ANALYSIS (simple mean, intersection of valid images)")
    print("=" * 80)
    print(f"{'Ensemble':<40} {'N':>5} {'overall':>10} {'sharpness':>10} {'color':>10} {'wSRCC':>10}")
    print("-" * 90)

    results_list = []
    for ens_name, model_names in ensembles.items():
        # Find common images
        common = set(gt.keys())
        for mn in model_names:
            common &= set(ensemble_models[mn].keys())
        common = sorted(common)

        if len(common) == 0:
            print(f"{ens_name:<40} {'NO COMMON IMAGES':>5}")
            continue

        # Compute ensemble predictions
        model_pred_list = [ensemble_models[mn] for mn in model_names]
        ens_pred = ensemble_preds(model_pred_list, common)
        m = compute_metrics(ens_pred, gt, common)
        results_list.append((ens_name, m))
        print(
            f"{ens_name:<40} {m['n']:>5} {m['overall']:>10.4f} {m['sharpness']:>10.4f} "
            f"{m['color_fidelity']:>10.4f} {m['wSRCC']:>10.4f}"
        )

    # Sort by wSRCC
    print("\n" + "=" * 80)
    print("RANKING BY wSRCC")
    print("=" * 80)
    results_list.sort(key=lambda x: x[1]["wSRCC"], reverse=True)
    for rank, (name, m) in enumerate(results_list, 1):
        print(f"  {rank}. {name:<38} wSRCC={m['wSRCC']:.4f}  (N={m['n']})")

    # Pairwise residual correlations (on common images for the 4 key models)
    key_models = ["Flash", "GPT-4.1", "Qwen122B", "Lite(comb)"]
    common_all = set(gt.keys())
    for mn in key_models:
        common_all &= set(ensemble_models[mn].keys())
    common_all = sorted(common_all)
    print(f"\nResidual correlations computed on {len(common_all)} common images")

    pairwise_residual_corr(
        {mn: ensemble_models[mn] for mn in key_models},
        gt,
        common_all,
    )

    # Also compare Qwen arm7 vs baseline and Lite arm10 vs baseline
    print("\n" + "=" * 80)
    print("PROMPT OPTIMIZATION LIFT (arm vs baseline)")
    print("=" * 80)
    comparisons = [
        ("Qwen122B(arm7)", "Qwen122B(base)"),
        ("Lite(arm10)", "Lite(base)"),
    ]
    for opt, base in comparisons:
        common = sorted(set(preds[opt].keys()) & set(preds[base].keys()) & set(gt.keys()))
        m_opt = compute_metrics(preds[opt], gt, common)
        m_base = compute_metrics(preds[base], gt, common)
        delta = m_opt["wSRCC"] - m_base["wSRCC"]
        print(f"\n  {opt} vs {base}  (N={len(common)})")
        print(f"    {'':>15} {'overall':>10} {'sharpness':>10} {'color':>10} {'wSRCC':>10}")
        print(
            f"    {'Optimized':<15} {m_opt['overall']:>10.4f} {m_opt['sharpness']:>10.4f} "
            f"{m_opt['color_fidelity']:>10.4f} {m_opt['wSRCC']:>10.4f}"
        )
        print(
            f"    {'Baseline':<15} {m_base['overall']:>10.4f} {m_base['sharpness']:>10.4f} "
            f"{m_base['color_fidelity']:>10.4f} {m_base['wSRCC']:>10.4f}"
        )
        print(f"    {'Delta':<15} {'':>10} {'':>10} {'':>10} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
