"""Evaluate VLM quality ratings against DIQA-5000 human MOS ground truth.

Supports two execution modes:
- **Local**: Reads images from a local directory, calls VLM APIs directly.
- **Modal**: Runs on Modal cloud infrastructure with GCS dataset download.

Usage (local):
    # Quick test with repo sample image
    python results/vlm_teacher_eval/run_eval.py --test --provider anthropic

    # Full evaluation against local DIQA-5000 test split
    python results/vlm_teacher_eval/run_eval.py \
        --data-dir /path/to/diqa5000 --split test --provider anthropic

    # Multi-pass variance estimation
    python results/vlm_teacher_eval/run_eval.py \
        --data-dir /path/to/diqa5000 --num-passes 5 --temperature 0.7

Usage (Modal):
    uv run modal run results/vlm_teacher_eval/run_eval.py --test
    uv run modal run --detach results/vlm_teacher_eval/run_eval.py

Purpose:
    Before investing in VLM-generated pseudo-labels for SigLIP2-IQA
    training data expansion, this script validates how well each VLM
    correlates with human perceptual quality judgments on the DIQA-5000
    benchmark per dimension (overall, sharpness, color fidelity).
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# IQA dimensions
# ---------------------------------------------------------------------------
IQA_DIMS = ("overall", "sharpness", "color")

# ---------------------------------------------------------------------------
# Try Modal import — gracefully degrade to local-only mode
# ---------------------------------------------------------------------------
try:
    import modal

    _HAS_MODAL = True
except ImportError:
    _HAS_MODAL = False

# ---------------------------------------------------------------------------
# Modal app definition (only when Modal is available)
# ---------------------------------------------------------------------------
if _HAS_MODAL:
    app = modal.App("vlm-diqa-correlation")

    results_volume = modal.Volume.from_name(
        "vlm-diqa-correlation-results", create_if_missing=True
    )
    diqa5000_volume = modal.Volume.from_name(
        "diqa5000-original", create_if_missing=True
    )

    eval_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "anthropic>=0.43.0",
        "openai>=1.0.0",
        "numpy<2.0",
        "Pillow>=11.0.0",
        "scipy",
        "tqdm",
    )

GCS_BUCKET = "image_detection_b"
GCS_PREFIX = "datasets/diqa-5000-original"
DIQA5000_SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class VLMEvalConfig:
    """Configuration for VLM correlation evaluation."""

    provider: str = "anthropic"
    model: str = ""  # auto-selected based on provider if empty
    max_tokens: int = 512
    temperature: float = 0.0
    num_passes: int = 1

    scale_min: float = 1.0
    scale_max: float = 5.0
    scale_step: float = 0.1

    split: str = "test"
    max_images: int | None = None
    image_max_pixels: int = 1024

    max_retries: int = 3
    retry_delay: float = 2.0
    requests_per_minute: int = 50
    batch_size: int = 10

    output_dir: str = "results/vlm_teacher_eval/output"
    resume: bool = False

    def resolve_model(self) -> str:
        """Return the model ID, auto-selecting based on provider if empty."""
        if self.model:
            return self.model
        defaults = {
            "anthropic": "claude-sonnet-4-6",
            "openrouter": "anthropic/claude-sonnet-4-6",
        }
        return defaults.get(self.provider, "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# GCS download (Modal mode)
# ---------------------------------------------------------------------------
def _setup_gcs_credentials() -> tuple[str | None, str | None]:
    """Write GCS credentials from Modal secret to temp file."""
    import tempfile

    prior = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if not gcp_sa_key:
        return None, prior

    sa_json = base64.b64decode(gcp_sa_key).decode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as cred_file:
        cred_file.write(sa_json)
        credentials_path = cred_file.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    return credentials_path, prior


def _download_diqa5000_from_gcs(data_dir: Path) -> bool:
    """Download DIQA-5000 dataset from GCS (cache-aware)."""
    from google.cloud import storage

    credentials_path, prior_creds = _setup_gcs_credentials()

    try:
        marker_file = data_dir / ".download_complete"
        if marker_file.exists():
            all_csvs = all(
                (data_dir / split / f"{split}.csv").exists()
                for split in DIQA5000_SPLITS
            )
            if all_csvs:
                print("DIQA-5000 already downloaded, skipping...")
                return True
            marker_file.unlink()

        print(f"Downloading DIQA-5000 from gs://{GCS_BUCKET}/{GCS_PREFIX}/")
        start_time = time.time()

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        data_dir.mkdir(parents=True, exist_ok=True)
        downloaded = 0

        for split in DIQA5000_SPLITS:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)
            (split_dir / "res").mkdir(exist_ok=True)

            prefix = f"{GCS_PREFIX}/{split}/"
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith("/"):
                    continue
                relative_path = blob.name[len(prefix) :]
                if not relative_path:
                    continue
                local_file = (split_dir / relative_path).resolve()
                if not str(local_file).startswith(
                    str(split_dir.resolve()) + "/"
                ):
                    print(f"  SKIPPING suspicious path: {blob.name!r}")
                    continue
                local_file.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_file))
                downloaded += 1
                if downloaded % 500 == 0:
                    print(f"  Downloaded {downloaded} files...")

        elapsed = time.time() - start_time
        print(f"Downloaded {downloaded} files in {elapsed:.1f}s")

        if downloaded < 100:
            print(f"ERROR: Only {downloaded} files downloaded")
            return False

        marker_file.touch()
        return True

    finally:
        if credentials_path:
            cred = Path(credentials_path)
            if cred.exists():
                cred.unlink()
        if prior_creds is not None:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = prior_creds
        elif credentials_path is not None:
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_diqa_samples(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Load DIQA-5000 samples with human MOS labels.

    Args:
        data_dir: Root directory containing split subdirectories.
        split: Dataset split (train, val, test).

    Returns:
        List of sample dicts with image_path, image_id, and MOS scores.
    """
    csv_path = data_dir / split / f"{split}.csv"
    res_dir = data_dir / split / "res"

    if not csv_path.exists():
        msg = f"CSV not found: {csv_path}"
        raise FileNotFoundError(msg)

    samples = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = res_dir / row["res"]
            if not image_path.exists():
                continue
            samples.append(
                {
                    "image_path": str(image_path),
                    "image_id": row["res"].replace(".jpg", ""),
                    "mos_overall": float(row["overall"]),
                    "mos_sharpness": float(row["sharpness"]),
                    "mos_color": float(row["color_fidelity"]),
                }
            )
    print(f"  {split}: {len(samples)} samples loaded")
    return samples


# ---------------------------------------------------------------------------
# Rating loop
# ---------------------------------------------------------------------------
def _create_client(config: VLMEvalConfig) -> Any:
    """Create the appropriate VLM client based on provider."""
    # Import here to allow the module to load without API SDKs installed
    from results.vlm_teacher_eval.vlm_client import (
        AnthropicClient,
        OpenRouterClient,
    )

    model = config.resolve_model()
    if config.provider == "anthropic":
        return AnthropicClient(model=model)
    elif config.provider == "openrouter":
        return OpenRouterClient(model=model)
    else:
        msg = f"Unknown provider: {config.provider}"
        raise ValueError(msg)


def rate_image_with_retries(
    client: Any,
    image_path: str,
    config: VLMEvalConfig,
    system_prompt: str,
    pass_idx: int = 0,
) -> dict[str, Any]:
    """Rate a single image with retry logic.

    Args:
        client: VLM client instance.
        image_path: Path to the image file.
        config: Evaluation configuration.
        system_prompt: Formatted system prompt.
        pass_idx: Pass index for multi-pass evaluation.

    Returns:
        Result dict with scores, raw response, latency, and error info.
    """
    from results.vlm_teacher_eval.image_utils import encode_image_base64
    from results.vlm_teacher_eval.prompts import USER_PROMPT
    from results.vlm_teacher_eval.response_parser import parse_iqa_response

    result: dict[str, Any] = {
        "pass_idx": pass_idx,
        "success": False,
        "error": None,
        "overall": None,
        "sharpness": None,
        "color_fidelity": None,
        "reasoning": None,
        "raw_response": None,
        "latency_ms": None,
    }

    temp = (
        config.temperature
        if config.num_passes == 1
        else max(config.temperature, 0.7)
    )

    for attempt in range(config.max_retries):
        try:
            b64_data, media_type = encode_image_base64(
                image_path, config.image_max_pixels
            )
            response = client.rate_image(
                image_b64=b64_data,
                media_type=media_type,
                system_prompt=system_prompt,
                user_prompt=USER_PROMPT,
                temperature=temp,
                max_tokens=config.max_tokens,
            )

            result["raw_response"] = response.text
            result["latency_ms"] = response.latency_ms

            rating = parse_iqa_response(
                response.text, config.scale_min, config.scale_max
            )
            result["overall"] = rating.overall
            result["sharpness"] = rating.sharpness
            result["color_fidelity"] = rating.color_fidelity
            result["reasoning"] = rating.reasoning
            result["success"] = True
            return result

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if attempt < config.max_retries - 1:
                delay = config.retry_delay * (2**attempt)
                print(
                    f"    Retry {attempt + 1}/{config.max_retries} "
                    f"after {delay:.1f}s: {result['error']}"
                )
                time.sleep(delay)

    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def compute_full_analysis(
    all_results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    config: VLMEvalConfig,
) -> dict[str, Any]:
    """Compute full correlation analysis across all dimensions and passes.

    Args:
        all_results: List of per-image rating results.
        samples: Original DIQA-5000 samples with human MOS.
        config: Evaluation configuration.

    Returns:
        Analysis dictionary with per-dimension correlations and VQualA.
    """
    import numpy as np

    from results.vlm_teacher_eval.correlation import compute_correlations

    dim_map = {
        "overall": "mos_overall",
        "sharpness": "mos_sharpness",
        "color": "mos_color",
    }
    vlm_key_map = {
        "overall": "overall",
        "sharpness": "sharpness",
        "color": "color_fidelity",
    }

    sample_lookup = {s["image_id"]: s for s in samples}

    analysis: dict[str, Any] = {
        "config": {
            "provider": config.provider,
            "model": config.resolve_model(),
            "split": config.split,
            "num_passes": config.num_passes,
            "temperature": config.temperature,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_images": len(samples),
        "successful_ratings": sum(1 for r in all_results if r["success"]),
        "failed_ratings": sum(1 for r in all_results if not r["success"]),
        "per_dimension": {},
        "vquala": None,
    }

    for dim in IQA_DIMS:
        vlm_key = vlm_key_map[dim]
        mos_key = dim_map[dim]

        image_vlm_scores: dict[str, list[float]] = {}
        for r in all_results:
            if r["success"] and r[vlm_key] is not None:
                img_id = r["image_id"]
                image_vlm_scores.setdefault(img_id, []).append(r[vlm_key])

        vlm_means = []
        mos_values = []
        vlm_stds = []

        for img_id, scores in image_vlm_scores.items():
            if img_id in sample_lookup:
                vlm_means.append(float(np.mean(scores)))
                vlm_stds.append(
                    float(np.std(scores)) if len(scores) > 1 else 0.0
                )
                mos_values.append(sample_lookup[img_id][mos_key])

        corr = compute_correlations(vlm_means, mos_values)

        if config.num_passes > 1 and vlm_stds:
            corr["inter_run_mean_std"] = float(np.mean(vlm_stds))
            corr["inter_run_max_std"] = float(np.max(vlm_stds))

        analysis["per_dimension"][dim] = corr

    # VQualA composite
    dims = analysis["per_dimension"]
    if all(dim in dims for dim in IQA_DIMS):
        vquala = (
            0.5 * dims["overall"]["srcc"]
            + 0.25 * dims["sharpness"]["srcc"]
            + 0.25 * dims["color"]["srcc"]
        )
        analysis["vquala"] = float(vquala)

    return analysis


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def save_results(
    output_dir: Path,
    all_results: list[dict[str, Any]],
    analysis: dict[str, Any],
    config: VLMEvalConfig,
) -> None:
    """Save raw results and analysis to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = config.resolve_model().split("/")[-1].replace("-", "_")

    # Raw per-image results
    raw_path = (
        output_dir
        / f"vlm_ratings_{model_short}_{config.split}_{timestamp}.jsonl"
    )
    with open(raw_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  Raw results: {raw_path}")

    # Analysis summary
    if analysis:
        analysis_path = (
            output_dir
            / f"correlation_analysis_{model_short}_{config.split}_{timestamp}.json"
        )
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"  Analysis: {analysis_path}")


def print_summary(analysis: dict[str, Any]) -> None:
    """Print a human-readable summary of the analysis."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Dimension':<12} {'SRCC':>8} {'PLCC':>8}")
    print("-" * 30)
    for dim in IQA_DIMS:
        d = analysis["per_dimension"].get(dim, {})
        print(f"{dim:<12} {d.get('srcc', 0):.4f}   {d.get('plcc', 0):.4f}")
    print(f"\n{'VQualA':<12} {analysis.get('vquala', 0):.4f}")
    print(f"\nRef: SigLIP2-IQA-Base VQualA = 0.886")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(config: VLMEvalConfig, data_dir: Path) -> dict[str, Any]:
    """Run VLM evaluation against DIQA-5000 human MOS ground truth.

    Args:
        config: Evaluation configuration.
        data_dir: Path to DIQA-5000 dataset root.

    Returns:
        Analysis dictionary.
    """
    from results.vlm_teacher_eval.prompts import build_system_prompt

    print("=" * 60)
    print("VLM DIQA-5000 Correlation Evaluation")
    print("=" * 60)
    print(f"Provider: {config.provider}")
    print(f"Model: {config.resolve_model()}")
    print(f"Split: {config.split}")
    print(f"Passes: {config.num_passes}")
    print(f"Temperature: {config.temperature}")
    print()

    # Load samples
    print(f"Loading {config.split} split...")
    samples = load_diqa_samples(data_dir, config.split)

    if config.max_images:
        samples = samples[: config.max_images]
        print(f"  Limited to {len(samples)} images")

    # Initialize
    client = _create_client(config)
    system_prompt = build_system_prompt()

    all_results: list[dict[str, Any]] = []
    total_tasks = len(samples) * config.num_passes
    completed = 0
    failed = 0
    start_time = time.time()
    request_times: list[float] = []

    print(
        f"\nRating {len(samples)} images x {config.num_passes} passes "
        f"= {total_tasks} total ratings\n"
    )

    for sample in samples:
        image_id = sample["image_id"]

        for pass_idx in range(config.num_passes):
            # Rate limiting
            now = time.time()
            request_times = [t for t in request_times if now - t < 60]
            if len(request_times) >= config.requests_per_minute:
                wait = 60 - (now - request_times[0]) + 0.5
                if wait > 0:
                    print(f"  Rate limit: waiting {wait:.1f}s...")
                    time.sleep(wait)

            result = rate_image_with_retries(
                client, sample["image_path"], config, system_prompt, pass_idx
            )
            result["image_id"] = image_id
            all_results.append(result)
            request_times.append(time.time())

            if result["success"]:
                completed += 1
            else:
                failed += 1
                print(
                    f"  FAILED [{image_id}] pass {pass_idx}: {result['error']}"
                )

            # Progress
            total_done = completed + failed
            if total_done % 25 == 0 or total_done == total_tasks:
                elapsed = time.time() - start_time
                rate = total_done / max(elapsed, 1) * 60
                eta = (total_tasks - total_done) / max(rate / 60, 0.001)
                print(
                    f"  Progress: {total_done}/{total_tasks} "
                    f"({completed} ok, {failed} fail) "
                    f"| {rate:.0f}/min | ETA: {eta:.0f}s"
                )

            # Periodic save
            output_dir = Path(config.output_dir)
            if total_done % 100 == 0 and total_done > 0:
                save_results(output_dir, all_results, {}, config)

    # Final analysis
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({completed} rated, {failed} failed)")

    analysis = compute_full_analysis(all_results, samples, config)
    save_results(Path(config.output_dir), all_results, analysis, config)
    print_summary(analysis)

    return analysis


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------
if _HAS_MODAL:

    @app.function(
        image=eval_image,
        gpu=None,
        timeout=7200,
        volumes={
            "/results": results_volume,
            "/data": diqa5000_volume,
        },
        secrets=[
            modal.Secret.from_name("anthropic-api-key"),
            modal.Secret.from_name("gcs-credentials"),
        ],
    )
    def modal_run_evaluation(
        test: bool = False,
        split: str = "test",
        num_passes: int = 1,
        temperature: float = 0.0,
        max_images: int | None = None,
        provider: str = "anthropic",
    ) -> dict[str, Any]:
        """Modal-hosted evaluation function."""
        config = VLMEvalConfig(
            provider=provider,
            split=split,
            num_passes=num_passes,
            temperature=temperature,
            max_images=max_images,
            output_dir="/results/vlm_correlation",
        )

        if test:
            config.max_images = 10
            config.num_passes = 1
            print("\n*** TEST MODE: 10 images, 1 pass ***\n")

        data_dir = Path("/data/diqa5000")
        print("Downloading DIQA-5000 from GCS...")
        if not _download_diqa5000_from_gcs(data_dir):
            msg = "Failed to download DIQA-5000"
            raise RuntimeError(msg)
        diqa5000_volume.commit()

        result = run_evaluation(config, data_dir)
        results_volume.commit()
        return result

    @app.local_entrypoint()
    def modal_main(
        test: bool = False,
        split: str = "test",
        num_passes: int = 1,
        temperature: float = 0.0,
        max_images: int | None = None,
        provider: str = "anthropic",
    ) -> None:
        """Modal CLI entrypoint."""
        result = modal_run_evaluation.remote(
            test=test,
            split=split,
            num_passes=num_passes,
            temperature=temperature,
            max_images=max_images,
            provider=provider,
        )
        print("\n" + json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Local CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Local CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLM IQA ratings against DIQA-5000 human MOS"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to DIQA-5000 dataset root",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openrouter"],
        help="VLM provider",
    )
    parser.add_argument("--model", default="", help="Model ID override")
    parser.add_argument(
        "--num-passes", type=int, default=1, help="Number of rating passes"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit number of images"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (10 images, 1 pass)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/vlm_teacher_eval/output",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

    config = VLMEvalConfig(
        provider=args.provider,
        model=args.model,
        split=args.split,
        num_passes=args.num_passes,
        temperature=args.temperature,
        max_images=args.max_images,
        output_dir=args.output_dir,
    )

    if args.test:
        config.max_images = 10
        config.num_passes = 1
        print("\n*** TEST MODE: 10 images, 1 pass ***\n")

    run_evaluation(config, args.data_dir)


if __name__ == "__main__":
    main()
