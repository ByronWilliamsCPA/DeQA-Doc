"""Zero-shot linear probe of SigLIP2 backbone variants on DIQA-5000.

Extracts frozen backbone embeddings from 11 SigLIP2 model variants and
evaluates how much quality-relevant structure each backbone encodes via
Ridge regression linear probes per quality dimension.

This answers: "Which backbone has the most fine-tuning headroom for DIQA?"
before investing training compute.

Detach-safe: all computation (extraction, linear probe, aggregation) runs
on Modal remote functions. Results are saved to a Modal volume so they
survive ``modal run --detach``.

Usage:
    # Run full extraction + probe (11 models, ~30 min on L4)
    uv run modal run modal/siglip2_backbone_probe.py

    # Single model (for debugging)
    uv run modal run modal/siglip2_backbone_probe.py --model google/siglip2-base-patch16-naflex

    # Detached (recommended for full run)
    uv run modal run --detach modal/siglip2_backbone_probe.py

    # Fetch results after a detached run completes
    uv run modal run modal/siglip2_backbone_probe.py --fetch-results

    # Dry run: verify volume paths, GT loading, and image access
    uv run modal run modal/siglip2_backbone_probe.py --dry-run
"""

from __future__ import annotations

import modal

# ============================================================================
# Modal App & Volumes
# ============================================================================

app = modal.App("siglip2-backbone-probe")

# Volume layout (verified via `modal volume ls`):
#   diqa5000-original: /diqa5000/{train,val,test}/{res/,ori/,*.csv}
#   diqa-test-data:    /test.csv, /images/test_res_*.jpg
diqa_volume = modal.Volume.from_name("diqa5000-original")
diqa_test_volume = modal.Volume.from_name("diqa-test-data")
results_volume = modal.Volume.from_name("siglip2-probe-results", create_if_missing=True)

RESULTS_VOL_PATH = "/probe_results"

# Meta JSONs are local (not on volumes), so we bundle them into the image.
META_DIR = "DeQA-Score/Data-DeQA-Score/DIQA/metas"

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "transformers>=4.51.0",
        "accelerate",
        "pillow",
        "numpy<2.0",
        "tqdm",
        "scikit-learn",
        "scipy",
    )
    .add_local_file(f"{META_DIR}/train_diqa_overall.json", "/metas/train_diqa_overall.json")
    .add_local_file(f"{META_DIR}/train_diqa_sharpness.json", "/metas/train_diqa_sharpness.json")
    .add_local_file(f"{META_DIR}/train_diqa_color.json", "/metas/train_diqa_color.json")
)

# Lightweight image for dry-run (no torch/GPU needed)
dryrun_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pillow")
    .add_local_file(f"{META_DIR}/train_diqa_overall.json", "/metas/train_diqa_overall.json")
    .add_local_file(f"{META_DIR}/train_diqa_sharpness.json", "/metas/train_diqa_sharpness.json")
    .add_local_file(f"{META_DIR}/train_diqa_color.json", "/metas/train_diqa_color.json")
)

# ============================================================================
# Model Variants
# ============================================================================

# Each entry: (model_id, is_naflex, description)
# Organized by what variable each isolates.
MODEL_VARIANTS: list[dict[str, str | bool]] = [
    # --- Anchor ---
    {
        "model_id": "google/siglip2-base-patch16-naflex",
        "naflex": True,
        "label": "base-p16-naflex",
        "tests": "Baseline (current student)",
    },
    # --- NaFlex vs Fixed @ Base ---
    {
        "model_id": "google/siglip2-base-patch16-384",
        "naflex": False,
        "label": "base-p16-384",
        "tests": "NaFlex vs fixed @ Base",
    },
    # --- Resolution scaling @ Base ---
    {
        "model_id": "google/siglip2-base-patch16-512",
        "naflex": False,
        "label": "base-p16-512",
        "tests": "Resolution ceiling @ Base",
    },
    # --- Coarse patch / ultra-light student ---
    {
        "model_id": "google/siglip2-base-patch32-256",
        "naflex": False,
        "label": "base-p32-256",
        "tests": "Coarse patch, ultra-light student",
    },
    # --- Scale: Large ---
    {
        "model_id": "google/siglip2-large-patch16-384",
        "naflex": False,
        "label": "large-p16-384",
        "tests": "Scale step (303M)",
    },
    # --- Scale + Resolution: Large ---
    {
        "model_id": "google/siglip2-large-patch16-512",
        "naflex": False,
        "label": "large-p16-512",
        "tests": "Scale + resolution (303M @ 512)",
    },
    # --- Primary teacher candidate ---
    {
        "model_id": "google/siglip2-so400m-patch16-naflex",
        "naflex": True,
        "label": "so400m-p16-naflex",
        "tests": "Primary teacher candidate (NaFlex)",
    },
    # --- Patch-14 (more tokens) ---
    {
        "model_id": "google/siglip2-so400m-patch14-384",
        "naflex": False,
        "label": "so400m-p14-384",
        "tests": "Patch-14 (729 tokens vs 576)",
    },
    # --- NaFlex vs Fixed @ So400m ---
    {
        "model_id": "google/siglip2-so400m-patch16-384",
        "naflex": False,
        "label": "so400m-p16-384",
        "tests": "NaFlex vs fixed @ So400m",
    },
    # --- Resolution ceiling @ So400m ---
    {
        "model_id": "google/siglip2-so400m-patch16-512",
        "naflex": False,
        "label": "so400m-p16-512",
        "tests": "Resolution ceiling @ So400m",
    },
    # --- Maximum capacity ---
    {
        "model_id": "google/siglip2-giant-opt-patch16-384",
        "naflex": False,
        "label": "giant-p16-384",
        "tests": "Maximum capacity (1B vision)",
    },
]


# ============================================================================
# Data loading helpers (run inside Modal container)
# ============================================================================


def _load_train_gt(meta_dir: str) -> list[dict]:
    """Load DIQA-5000 train GT from per-dimension meta JSONs.

    Returns list of dicts with keys: image_path, overall, sharpness, color.
    """
    import json
    from pathlib import Path

    meta = Path(meta_dir)
    overall = json.loads((meta / "train_diqa_overall.json").read_text())
    sharpness = json.loads((meta / "train_diqa_sharpness.json").read_text())
    color = json.loads((meta / "train_diqa_color.json").read_text())

    sharp_by_id = {s["id"]: s["gt_score"] for s in sharpness}
    color_by_id = {s["id"]: s["gt_score"] for s in color}

    samples = []
    for item in overall:
        img_id = item["id"]
        if img_id in sharp_by_id and img_id in color_by_id:
            samples.append({
                "image_path": item["image"],
                "overall": item["gt_score"],
                "sharpness": sharp_by_id[img_id],
                "color": color_by_id[img_id],
            })
    return samples


def _load_test_gt(csv_path: str) -> list[dict]:
    """Load DIQA-5000 test GT from CSV.

    Returns list of dicts with keys: image_name, overall, sharpness, color.
    """
    import csv
    from pathlib import Path

    samples = []
    with Path(csv_path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "image_name": row["res"],
                "overall": float(row["overall"]),
                "sharpness": float(row["sharpness"]),
                "color": float(row["color_fidelity"]),
            })
    return samples


# ============================================================================
# Dry-run: verify volume layout, GT loading, and image accessibility
# ============================================================================


@app.function(
    image=dryrun_image,
    timeout=300,
    volumes={
        "/diqa": diqa_volume,
        "/test_data": diqa_test_volume,
    },
)
def dry_run_check() -> dict:
    """Verify volume paths, GT files, and image accessibility.

    Checks:
        1. Train volume: meta JSON paths and image directory
        2. Test volume: CSV path and image directory
        3. Loads GT and verifies record counts
        4. Opens first + last image from each split to confirm readability
        5. Reports image dimensions for aspect ratio sanity check

    Returns:
        Dict with status, found paths, record counts, and sample image info.
    """
    import csv
    import json
    from pathlib import Path

    report: dict = {"status": "ok", "errors": [], "warnings": []}

    # --- 1. Check meta JSONs (bundled in container image at /metas/) ---
    print("=" * 60)
    print("DRY RUN: Checking volume layout and data access")
    print("=" * 60)

    print("\n--- Meta JSONs (/metas/, bundled in image) ---")
    meta_dir = Path("/metas")
    expected_files = [
        "train_diqa_overall.json",
        "train_diqa_sharpness.json",
        "train_diqa_color.json",
    ]
    for fname in expected_files:
        fpath = meta_dir / fname
        if fpath.exists():
            data = json.loads(fpath.read_text())
            print(f"  {fname}: {len(data)} records")
            report[f"train_{fname}_count"] = len(data)
            if data:
                print(f"    Keys: {list(data[0].keys())}")
                print(f"    Sample image_path: {data[0].get('image', 'N/A')}")
        else:
            report["errors"].append(f"Missing bundled meta: {fpath}")

    report["train_meta_dir"] = str(meta_dir)

    # --- 2. Check train images volume (diqa5000-original) ---
    print("\n--- Train Volume (/diqa/diqa5000/train/res/) ---")
    train_img_dir = Path("/diqa/diqa5000/train/res")
    if not train_img_dir.exists():
        # Try alternate layouts
        candidates = [
            Path("/diqa/train/res"),
            Path("/diqa/diqa5000/train"),
            Path("/diqa"),
        ]
        for c in candidates:
            if c.exists():
                print(f"  Expected path missing, checking {c}...")
                contents = sorted(p.name for p in c.iterdir())[:10]
                print(f"    Contents: {contents}")
        report["errors"].append(
            f"Train image dir not found at {train_img_dir}"
        )
    else:
        n_images = len(list(train_img_dir.glob("*.jpg")))
        print(f"  Image dir: {train_img_dir}")
        print(f"  JPG count: {n_images}")
        report["train_image_dir"] = str(train_img_dir)
        report["train_image_count"] = n_images

        # Check path resolution: meta says "DIQA/train/res/train_res_00001.jpg"
        # but images are at /diqa/diqa5000/train/res/train_res_00001.jpg
        # We need to resolve by basename.
        overall = json.loads((meta_dir / "train_diqa_overall.json").read_text())
        if overall:
            sample_path = overall[0]["image"]  # e.g. "DIQA/train/res/train_res_00001.jpg"
            basename = Path(sample_path).name
            resolved_by_name = train_img_dir / basename
            print(f"  Meta image_path: '{sample_path}'")
            print(f"    Basename resolve: {resolved_by_name} -> {resolved_by_name.exists()}")
            report["train_path_resolution"] = (
                "basename" if resolved_by_name.exists() else "FAILED"
            )
            if not resolved_by_name.exists():
                report["errors"].append(
                    f"Cannot resolve train image '{basename}' in {train_img_dir}"
                )

        # Open sample images
        from PIL import Image

        samples = sorted(train_img_dir.glob("*.jpg"))
        for img_path in [samples[0], samples[-1]] if len(samples) >= 2 else samples:
            img = Image.open(img_path)
            print(f"  Sample: {img_path.name} -> {img.size[0]}x{img.size[1]} {img.mode}")
            report.setdefault("train_sample_images", []).append({
                "name": img_path.name,
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "aspect_ratio_hw": round(img.size[1] / img.size[0], 3),
            })

    # Also check val split
    print("\n--- Val Volume (/diqa/diqa5000/val/res/) ---")
    val_img_dir = Path("/diqa/diqa5000/val/res")
    if val_img_dir.exists():
        n_val = len(list(val_img_dir.glob("*.jpg")))
        print(f"  Val images: {n_val}")
        report["val_image_count"] = n_val
    else:
        print("  Val dir not found (optional, not blocking)")

    # --- 2. Discover test volume layout ---
    print("\n--- Test Volume (/test_data) ---")
    test_root = Path("/test_data")
    if not test_root.exists():
        report["errors"].append("Test volume not mounted at /test_data")
        report["status"] = "FAIL"
        return report

    top_level_test = sorted(p.name for p in test_root.iterdir())
    print(f"  Top-level entries: {top_level_test[:20]}")
    report["test_top_level"] = top_level_test[:20]

    # Find test.csv
    csv_candidates = [
        test_root / "test.csv",
        test_root / "data" / "test.csv",
    ]
    test_csv = None
    for candidate in csv_candidates:
        if candidate.exists():
            test_csv = candidate
            break

    if test_csv is None:
        found_csv = list(test_root.rglob("test.csv"))
        if found_csv:
            test_csv = found_csv[0]

    if test_csv:
        with test_csv.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"  test.csv: {test_csv} -> {len(rows)} records")
        print(f"    Columns: {list(rows[0].keys()) if rows else 'N/A'}")
        report["test_csv_path"] = str(test_csv)
        report["test_csv_count"] = len(rows)
        report["test_csv_columns"] = list(rows[0].keys()) if rows else []

        if rows:
            print(f"    First image: {rows[0].get('res', 'N/A')}")
    else:
        report["errors"].append("Cannot find test.csv")

    # Find test image directory
    print("\n  Searching for test images...")
    test_img_candidates = [
        test_root / "images",
        test_root / "res",
        test_root,
    ]
    test_img_dir = None
    for candidate in test_img_candidates:
        if candidate.exists() and list(candidate.glob("test_res_*.jpg")):
            test_img_dir = candidate
            break

    if test_img_dir is None:
        jpgs = list(test_root.rglob("test_res_00001.jpg"))
        if jpgs:
            test_img_dir = jpgs[0].parent

    if test_img_dir:
        n_test_images = len(list(test_img_dir.glob("*.jpg")))
        print(f"  Image dir: {test_img_dir}")
        print(f"  JPG count: {n_test_images}")
        report["test_image_dir"] = str(test_img_dir)
        report["test_image_count"] = n_test_images

        # Open sample images
        from PIL import Image

        samples = sorted(test_img_dir.glob("*.jpg"))
        for img_path in [samples[0], samples[-1]] if len(samples) >= 2 else samples:
            img = Image.open(img_path)
            print(f"  Sample: {img_path.name} -> {img.size[0]}x{img.size[1]} {img.mode}")
            report.setdefault("test_sample_images", []).append({
                "name": img_path.name,
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
                "aspect_ratio_hw": round(img.size[1] / img.size[0], 3),
            })

        # Verify CSV image names match actual files
        if test_csv and rows:
            csv_names = {row["res"] for row in rows}
            dir_names = {p.name for p in test_img_dir.glob("*.jpg")}
            matched = len(csv_names & dir_names)
            csv_only = len(csv_names - dir_names)
            dir_only = len(dir_names - csv_names)
            print(f"  CSV↔Dir match: {matched} matched, {csv_only} CSV-only, {dir_only} dir-only")
            report["test_csv_image_match"] = matched
            if csv_only > 0:
                report["warnings"].append(
                    f"{csv_only} test images in CSV not found on disk"
                )
    else:
        report["errors"].append("Cannot find test image directory")

    # --- 3. Summary ---
    print(f"\n{'=' * 60}")
    if report["errors"]:
        report["status"] = "FAIL"
        print(f"DRY RUN: FAIL ({len(report['errors'])} errors)")
        for e in report["errors"]:
            print(f"  ERROR: {e}")
    else:
        print("DRY RUN: OK — all paths verified")

    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARN: {w}")

    # Print the exact paths the extraction function should use
    print("\n--- Resolved Paths for Extraction ---")
    print(f"  train_meta_dir: {report.get('train_meta_dir', 'NOT FOUND')}")
    print(f"  train_image_dir: {report.get('train_image_dir', 'NOT FOUND')}")
    print(f"  train_path_resolution: {report.get('train_path_resolution', 'UNKNOWN')}")
    print(f"  test_csv_path: {report.get('test_csv_path', 'NOT FOUND')}")
    print(f"  test_image_dir: {report.get('test_image_dir', 'NOT FOUND')}")

    return report


# ============================================================================
# Embedding extraction (GPU)
# ============================================================================


@app.function(
    image=gpu_image,
    gpu="L4",
    timeout=3600,
    volumes={
        "/diqa": diqa_volume,
        "/test_data": diqa_test_volume,
    },
)
def extract_and_probe(model_id: str, is_naflex: bool) -> dict:
    """Extract frozen embeddings and run linear probe for one SigLIP2 variant.

    All computation (extraction + Ridge regression probe) happens on the
    remote GPU container so this function is detach-safe. Returns only
    probe metrics, not raw embeddings.

    Args:
        model_id: HuggingFace model ID (e.g., ``google/siglip2-base-patch16-naflex``).
        is_naflex: Whether this is a NaFlex variant (needs spatial_shapes).

    Returns:
        Dict with per-dimension SRCC/PLCC/MAE, wSRCC, and timing info.
    """
    import time

    import numpy as np
    import torch
    from PIL import Image
    from scipy import stats
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    from transformers import AutoModel, AutoProcessor

    print(f"\n{'=' * 60}")
    print(f"Extracting + probing: {model_id}")
    print(f"NaFlex: {is_naflex}")
    print(f"{'=' * 60}")

    # --- Load model ---
    t0 = time.time()
    model = AutoModel.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = model.to("cuda").eval()
    embed_dim = model.config.vision_config.hidden_size
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s (embed_dim={embed_dim})")

    # --- Helper: extract embedding from one image ---
    def get_embedding(pil_img: Image.Image) -> np.ndarray:
        if is_naflex:
            inputs = processor(
                images=pil_img,
                return_tensors="pt",
                max_num_patches=784,
                padding="max_length",
            )
            pixel_values = inputs["pixel_values"].to("cuda")
            spatial_shapes = inputs["spatial_shapes"].to("cuda")
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                out = model.get_image_features(
                    pixel_values=pixel_values,
                    spatial_shapes=spatial_shapes,
                )
        else:
            inputs = processor(
                images=pil_img,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to("cuda")
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                out = model.get_image_features(pixel_values=pixel_values)

        # Handle different return types
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            feat = out.last_hidden_state.mean(dim=1)
        elif isinstance(out, torch.Tensor):
            if out.dim() == 3:
                feat = out.mean(dim=1)
            else:
                feat = out
        else:
            msg = f"Unexpected output type: {type(out)}"
            raise TypeError(msg)

        return feat.cpu().float().numpy().squeeze(0)

    # --- Load ground truth ---
    from pathlib import Path

    train_gt = _load_train_gt("/metas")
    test_gt = _load_test_gt("/test_data/test.csv")
    print(f"Train GT: {len(train_gt)} samples, Test GT: {len(test_gt)} samples")

    train_img_dir = Path("/diqa/diqa5000/train/res")

    # --- Extract train embeddings ---
    print("\nExtracting train embeddings...")
    t1 = time.time()
    train_embeddings = []
    train_scores = {"overall": [], "sharpness": [], "color": []}
    skipped_train = 0

    for sample in tqdm(train_gt, desc="Train"):
        img_path = train_img_dir / Path(sample["image_path"]).name
        if not img_path.exists():
            skipped_train += 1
            continue

        pil_img = Image.open(img_path).convert("RGB")
        emb = get_embedding(pil_img)
        train_embeddings.append(emb)
        for dim in ("overall", "sharpness", "color"):
            train_scores[dim].append(sample[dim])

    train_time = time.time() - t1
    print(
        f"Train: {len(train_embeddings)} embeddings in {train_time:.1f}s "
        f"({skipped_train} skipped)"
    )

    # --- Extract test embeddings ---
    print("\nExtracting test embeddings...")
    t2 = time.time()
    test_embeddings = []
    test_scores = {"overall": [], "sharpness": [], "color": []}
    skipped_test = 0

    for sample in tqdm(test_gt, desc="Test"):
        img_path = Path("/test_data/images") / sample["image_name"]
        if not img_path.exists():
            alt = Path("/test_data") / sample["image_name"]
            if alt.exists():
                img_path = alt
            else:
                skipped_test += 1
                continue

        pil_img = Image.open(img_path).convert("RGB")
        emb = get_embedding(pil_img)
        test_embeddings.append(emb)
        for dim in ("overall", "sharpness", "color"):
            test_scores[dim].append(sample[dim])

    test_time = time.time() - t2
    print(
        f"Test: {len(test_embeddings)} embeddings in {test_time:.1f}s "
        f"({skipped_test} skipped)"
    )

    # --- Linear probe (Ridge regression) ---
    print("\nRunning linear probe...")
    X_train = np.stack(train_embeddings).astype(np.float32)
    X_test = np.stack(test_embeddings).astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    dimensions = ("overall", "sharpness", "color")
    dim_metrics: dict[str, dict[str, float]] = {}

    for dim in dimensions:
        y_train = np.array(train_scores[dim], dtype=np.float32)
        y_test = np.array(test_scores[dim], dtype=np.float32)

        best_srcc = -1.0
        best_alpha = 1.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train)
            pred = ridge.predict(X_test)
            srcc = float(stats.spearmanr(pred, y_test).statistic)
            if srcc > best_srcc:
                best_srcc = srcc
                best_alpha = alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)

        srcc = float(stats.spearmanr(pred, y_test).statistic)
        plcc = float(stats.pearsonr(pred, y_test).statistic)
        mae = float(np.mean(np.abs(pred - y_test)))

        dim_metrics[dim] = {
            "srcc": srcc,
            "plcc": plcc,
            "mae": mae,
            "best_alpha": best_alpha,
        }
        print(f"  {dim}: SRCC={srcc:.4f} PLCC={plcc:.4f} MAE={mae:.4f} alpha={best_alpha}")

    wsrcc = (
        0.5 * dim_metrics["overall"]["srcc"]
        + 0.25 * dim_metrics["sharpness"]["srcc"]
        + 0.25 * dim_metrics["color"]["srcc"]
    )
    print(f"  wSRCC={wsrcc:.4f}")

    return {
        "model_id": model_id,
        "embed_dim": embed_dim,
        "n_train": len(train_embeddings),
        "n_test": len(test_embeddings),
        "skipped_train": skipped_train,
        "skipped_test": skipped_test,
        "dimensions": dim_metrics,
        "wsrcc": wsrcc,
        "model_load_time_s": load_time,
        "extract_time_s": train_time + test_time,
    }


# ============================================================================
# Remote orchestrator (detach-safe)
# ============================================================================


def _format_results(all_probes: list[dict]) -> str:
    """Format probe results into printable tables and analysis.

    Args:
        all_probes: List of probe result dicts, sorted by wSRCC descending.

    Returns:
        Formatted string with all tables and decision matrix.
    """
    lines: list[str] = []

    # Summary table
    lines.append(f"\n{'=' * 100}")
    lines.append("RESULTS: SigLIP2 Zero-Shot Linear Probe on DIQA-5000")
    lines.append(f"{'=' * 100}")
    lines.append(
        f"| {'Rank':>4s} | {'Label':<22s} | {'Dim':>4s} | {'wSRCC':>7s} "
        f"| {'Overall':>7s} | {'Sharp':>7s} | {'Color':>7s} "
        f"| {'Time':>5s} | {'Tests':<35s} |"
    )
    lines.append(
        f"|{'-' * 6}|{'-' * 24}|{'-' * 6}|{'-' * 9}"
        f"|{'-' * 9}|{'-' * 9}|{'-' * 9}|{'-' * 7}|{'-' * 37}|"
    )

    for rank, p in enumerate(all_probes, 1):
        lines.append(
            f"| {rank:>4d} | {p['label']:<22s} | {p['embed_dim']:>4d} "
            f"| {p['wsrcc']:>7.4f} | {p['dimensions']['overall']['srcc']:>7.4f} "
            f"| {p['dimensions']['sharpness']['srcc']:>7.4f} "
            f"| {p['dimensions']['color']['srcc']:>7.4f} "
            f"| {p['extract_time_s']:>5.0f}s "
            f"| {p['tests']:<35s} |"
        )

    # Delta table vs baseline
    baseline = next(
        (p for p in all_probes if p["label"] == "base-p16-naflex"), None
    )
    if baseline:
        lines.append(f"\n{'=' * 80}")
        lines.append("DELTAS vs baseline (base-p16-naflex)")
        lines.append(f"{'=' * 80}")
        lines.append(
            f"| {'Label':<22s} | {'dwSRCC':>8s} | {'dOverall':>8s} "
            f"| {'dSharp':>8s} | {'dColor':>8s} |"
        )
        lines.append(f"|{'-' * 24}|{'-' * 10}|{'-' * 10}|{'-' * 10}|{'-' * 10}|")

        for p in all_probes:
            d_wsrcc = p["wsrcc"] - baseline["wsrcc"]
            d_o = p["dimensions"]["overall"]["srcc"] - baseline["dimensions"]["overall"]["srcc"]
            d_s = p["dimensions"]["sharpness"]["srcc"] - baseline["dimensions"]["sharpness"]["srcc"]
            d_c = p["dimensions"]["color"]["srcc"] - baseline["dimensions"]["color"]["srcc"]
            lines.append(
                f"| {p['label']:<22s} | {d_wsrcc:>+8.4f} | {d_o:>+8.4f} "
                f"| {d_s:>+8.4f} | {d_c:>+8.4f} |"
            )

    # Key comparisons
    lines.append(f"\n{'=' * 80}")
    lines.append("KEY COMPARISONS")
    lines.append(f"{'=' * 80}")

    def _get(label: str) -> dict | None:
        return next((p for p in all_probes if p["label"] == label), None)

    comparisons = [
        ("NaFlex vs Fixed @ Base", "base-p16-naflex", "base-p16-384"),
        ("NaFlex vs Fixed @ So400m", "so400m-p16-naflex", "so400m-p16-384"),
        ("Resolution 384->512 @ Base", "base-p16-384", "base-p16-512"),
        ("Resolution 384->512 @ So400m", "so400m-p16-384", "so400m-p16-512"),
        ("Patch-14 vs Patch-16 @ So400m", "so400m-p14-384", "so400m-p16-384"),
        ("Scale: Base -> Large", "base-p16-384", "large-p16-384"),
        ("Scale: Large -> So400m", "large-p16-384", "so400m-p16-384"),
        ("Scale: So400m -> Giant", "so400m-p16-384", "giant-p16-384"),
    ]

    for desc, label_a, label_b in comparisons:
        a = _get(label_a)
        b = _get(label_b)
        if a and b:
            delta = a["wsrcc"] - b["wsrcc"]
            delta_s = a["dimensions"]["sharpness"]["srcc"] - b["dimensions"]["sharpness"]["srcc"]
            winner = label_a if delta > 0 else label_b
            lines.append(
                f"  {desc}: {label_a} vs {label_b}  "
                f"dwSRCC={delta:+.4f} dSharp={delta_s:+.4f} -> {winner}"
            )

    # Decision matrix
    lines.append(f"\n{'=' * 80}")
    lines.append("DECISION MATRIX")
    lines.append(f"{'=' * 80}")

    if baseline:
        best = all_probes[0]
        gap = best["wsrcc"] - baseline["wsrcc"]

        if gap < 0.01:
            lines.append(
                "  RESULT: All variants within 1% of Base. "
                "Backbone is NOT the bottleneck -- invest in data expansion."
            )
        elif gap < 0.03:
            lines.append(
                f"  RESULT: Modest gain ({gap:+.4f} wSRCC) from {best['label']}. "
                "Teacher-student may help but data expansion is likely higher ROI."
            )
        else:
            lines.append(
                f"  RESULT: Significant gain ({gap:+.4f} wSRCC) from {best['label']}. "
                "Teacher-student pipeline is justified."
            )

        best_sharp = max(all_probes, key=lambda p: p["dimensions"]["sharpness"]["srcc"])
        sharp_gap = (
            best_sharp["dimensions"]["sharpness"]["srcc"]
            - baseline["dimensions"]["sharpness"]["srcc"]
        )
        if sharp_gap > 0.02:
            lines.append(
                f"  SHARPNESS: {best_sharp['label']} gains {sharp_gap:+.4f} on sharpness. "
                "Larger backbone resolves text-edge features better."
            )
        else:
            lines.append(
                f"  SHARPNESS: Gap is only {sharp_gap:+.4f}. "
                "Sharpness bottleneck is resolution/data, not backbone capacity."
            )

    return "\n".join(lines)


@app.function(
    image=gpu_image,
    timeout=7200,
    volumes={
        RESULTS_VOL_PATH: results_volume,
    },
)
def run_all_probes(model_ids: list[str], naflex_flags: list[str], labels: list[str], tests_list: list[str]) -> dict:
    """Orchestrate all probe jobs and save results to a Modal volume.

    This function runs remotely on Modal, so it survives ``--detach``.
    It calls ``extract_and_probe.map()`` to fan out GPU work, collects
    metrics, formats tables, and persists results JSON to the volume.

    Args:
        model_ids: HuggingFace model IDs to evaluate.
        naflex_flags: Per-model NaFlex booleans (as strings for map compat).
        labels: Per-model short labels.
        tests_list: Per-model test descriptions.

    Returns:
        Dict with all probe results and formatted report.
    """
    import json
    from pathlib import Path

    n = len(model_ids)
    print(f"\n{'=' * 70}")
    print(f"Orchestrator: launching {n} extract+probe jobs...")
    print(f"{'=' * 70}")

    # Fan out GPU work
    naflex_bools = [f == "True" for f in naflex_flags]
    results_iter = extract_and_probe.map(model_ids, naflex_bools)

    # Build label/tests lookup
    label_map = dict(zip(model_ids, labels))
    tests_map = dict(zip(model_ids, tests_list))

    all_probes: list[dict] = []
    for result in results_iter:
        mid = result["model_id"]
        result["label"] = label_map[mid]
        result["tests"] = tests_map[mid]
        all_probes.append(result)
        print(
            f"  {result['label']}: wSRCC={result['wsrcc']:.4f}  "
            f"O={result['dimensions']['overall']['srcc']:.4f}  "
            f"S={result['dimensions']['sharpness']['srcc']:.4f}  "
            f"C={result['dimensions']['color']['srcc']:.4f}"
        )

    # Sort by wSRCC descending
    all_probes.sort(key=lambda p: p["wsrcc"], reverse=True)

    # Format and print report
    report_text = _format_results(all_probes)
    print(report_text)

    # Save to Modal volume for retrieval after detach
    output_dir = Path(RESULTS_VOL_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "probe_results.json"
    results_path.write_text(json.dumps(all_probes, indent=2, default=str))

    report_path = output_dir / "probe_report.txt"
    report_path.write_text(report_text)

    results_volume.commit()
    print(f"\nResults saved to volume: {results_path}")
    print(f"Report saved to volume: {report_path}")

    return {
        "status": "ok",
        "n_models": len(all_probes),
        "results": all_probes,
        "report": report_text,
    }


# ============================================================================
# Local entrypoint
# ============================================================================


@app.local_entrypoint()
def main(
    model: str | None = None,
    dry_run: bool = False,
    fetch_results: bool = False,
):
    """Run backbone probe (detach-safe).

    All heavy computation runs on Modal remote functions. Use ``--detach``
    for fire-and-forget; retrieve results later with ``--fetch-results``.

    Args:
        model: Optional single model ID to test (default: all 11 variants).
        dry_run: If True, only verify volume paths and data access.
        fetch_results: If True, download results from the Modal volume.
    """
    import json
    from pathlib import Path

    # --- Fetch results mode ---
    if fetch_results:
        import subprocess

        output_dir = Path("results/siglip2_backbone_probe")
        output_dir.mkdir(parents=True, exist_ok=True)

        for fname in ("probe_results.json", "probe_report.txt"):
            print(f"Downloading {fname} from volume...")
            subprocess.run(
                [
                    "modal", "volume", "get",
                    "siglip2-probe-results", fname,
                    str(output_dir / fname),
                ],
                check=True,
            )

        # Print the report
        report_path = output_dir / "probe_report.txt"
        if report_path.exists():
            print(report_path.read_text())

        results_path = output_dir / "probe_results.json"
        if results_path.exists():
            print(f"\nResults saved to {results_path}")
        return

    # --- Dry run mode ---
    if dry_run:
        print("=" * 70)
        print("DRY RUN: Verifying volume layout and data accessibility")
        print("=" * 70)
        report = dry_run_check.remote()

        if report["status"] != "ok":
            print("\nFIXUP NEEDED:")
            if report.get("train_path_resolution") == "FAILED":
                meta_dir = report.get("train_meta_dir", "?")
                img_dir = report.get("train_image_dir", "?")
                print(
                    f"  Train image paths in meta JSONs don't resolve.\n"
                    f"  Meta dir: {meta_dir}\n"
                    f"  Image dir: {img_dir}\n"
                    f"  Update extract_and_probe() path resolution."
                )
        else:
            print("\nAll checks passed. Ready to run full extraction.")

        output_dir = Path("results/siglip2_backbone_probe")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dry_run_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"Report saved to {report_path}")
        return

    # --- Full run (detach-safe) ---
    if model:
        variants = [v for v in MODEL_VARIANTS if v["model_id"] == model]
        if not variants:
            print(f"ERROR: Unknown model '{model}'")
            print("Available models:")
            for v in MODEL_VARIANTS:
                print(f"  {v['model_id']}")
            return
    else:
        variants = MODEL_VARIANTS

    print("=" * 70)
    print("SigLIP2 Backbone Linear Probe — DIQA-5000 (detach-safe)")
    print(f"Models to evaluate: {len(variants)}")
    print("=" * 70)

    # Prepare arguments for the remote orchestrator
    model_ids = [str(v["model_id"]) for v in variants]
    naflex_flags = [str(v["naflex"]) for v in variants]
    labels = [str(v["label"]) for v in variants]
    tests_list = [str(v["tests"]) for v in variants]

    # This single remote call is the "last triggered function" —
    # it survives --detach because Modal keeps it running.
    print("\nLaunching remote orchestrator (detach-safe)...")
    result = run_all_probes.remote(model_ids, naflex_flags, labels, tests_list)

    # If we're still alive (not detached), print + save locally
    print(result["report"])

    output_dir = Path("results/siglip2_backbone_probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "probe_results.json"
    output_path.write_text(json.dumps(result["results"], indent=2, default=str))
    print(f"\nResults saved to {output_path}")
