"""Stream 3: Two-model VLM consensus labeling for real document datasets.

Uses a VLM ensemble (Gemini Flash Lite + Qwen 3.5-122B via OpenRouter) to
generate quality labels for real document images that lack human quality
annotations. Training weight = 0.5 (lower than deterministic streams).

Source datasets (all on E: drive, split-aware):
    OHR-Bench     700  (train only — val/test reserved)
    Tobacco800    500  (all — no reserved splits)
    SmartDoc-QA   500  (train only — val/test reserved; filename-based split)
    RealDAE       400  (all — no reserved splits; _in images only)
    FUNSD+        300  (train only — test reserved; filename-based split)
    OCR-Quality   200  (all — no reserved splits)
    SROIE         100  (train only — test reserved)
    Total:      2,700 samples

VLM consensus protocol:
    1. Both models agree (|Δ| ≤ 1.0 MOS) → use mean as label
    2. Models disagree (|Δ| > 1.0 MOS) → invoke tiebreaker (GPT-4.1)
    3. All three disagree → exclude sample
    4. Parse failure → retry once, then skip

Checkpointing: saves progress to vlm_checkpoint.jsonl after each image,
so the pipeline can resume after interruption.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .iqa_to_deqa import vlm_scores_to_deqa_record

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

VLM_WEIGHT = 0.5

# OpenRouter model IDs
PRIMARY_MODEL = "google/gemini-2.0-flash-lite-001"
SECONDARY_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
TIEBREAKER_MODEL = "openai/gpt-4.1"

# Quality level mapping (matches DeQA convention)
QUALITY_LEVEL_MAP: dict[str, float] = {
    "excellent": 5.0,
    "good": 4.0,
    "fair": 3.0,
    "poor": 2.0,
    "bad": 1.0,
}

QUALITY_PROMPT = (
    "Rate the overall quality of this document image. "
    "Consider readability, clarity, and general visual quality. "
    "Choose exactly one: excellent, good, fair, poor, or bad. "
    "Respond with only one word."
)

# Consensus thresholds
AGREEMENT_THRESHOLD = 1.0  # MOS units — within this = agreement
RETRY_LIMIT = 1  # retries on parse failure

# E: drive base paths
_E_BASE = Path("/mnt/e/image_detection")
_E_01 = _E_BASE / "01_base_data"
_E_02 = _E_BASE / "02_benchmark_only"


# ── Source Dataset Definitions ───────────────────────────────────────────────


@dataclass(frozen=True)
class SourceDataset:
    """Configuration for a real document dataset to label via VLM."""

    name: str
    image_dir: Path
    count: int
    has_text_gt: bool = False
    glob_patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg")
    split_filter: str | None = None  # "train" filename prefix filter
    exclude_suffix: str | None = None  # e.g. "_gt" to skip GT pair images
    recursive: bool = False
    split_manifest: Path | None = None  # JSON with train_images list
    description: str = ""


# SmartDoc-QA split manifest (document-level stratified 70/10/20)
_SMARTDOC_SPLIT_MANIFEST = (
    _E_02 / "smartdoc-qa" / "splits" / "smartdoc_qa_splits.json"
)

TIER1_SOURCES = [
    SourceDataset(
        name="tobacco800",
        image_dir=_E_01 / "degraded" / "tobacco800" / "images",
        count=700,
        has_text_gt=False,
        description="Tobacco800: archival scanned documents with real degradation",
    ),
    SourceDataset(
        name="smartdoc_qa",
        image_dir=_E_02 / "smartdoc-qa" / "Dataset SmartDoc-QA" / "Captured_Images",
        count=4260,  # Label ALL images for full VLM coverage
        has_text_gt=True,
        recursive=True,
        # No split_manifest filter — label everything, filter at training time
        description="SmartDoc-QA: all images VLM-labeled, train docs used for Tier 1",
    ),
    SourceDataset(
        name="realdae",
        image_dir=_E_01 / "camera_captured" / "realdae",
        count=500,
        has_text_gt=False,
        exclude_suffix="_gt",
        recursive=True,
        glob_patterns=("*_in.jpg", "*_in.png"),
        description="RealDAE: camera-captured input images (not GT pairs)",
    ),
    SourceDataset(
        name="funsd",
        image_dir=_E_01 / "forms" / "funsd",
        count=199,  # Label ALL (149 train + 50 test)
        has_text_gt=True,
        recursive=True,
        glob_patterns=("train/images/*.png", "test/images/*.png"),
        description="FUNSD: all images VLM-labeled, train/test by directory",
    ),
    SourceDataset(
        name="funsd_plus",
        image_dir=_E_01 / "forms" / "funsd_plus" / "images",
        count=1139,  # Label ALL (1,026 train + 113 test)
        has_text_gt=True,
        # No split_filter — label everything, tag split by filename prefix
        description="FUNSD+: all images VLM-labeled, train/test by filename prefix",
    ),
    SourceDataset(
        name="ocr_quality",
        image_dir=_E_01 / "ocr_quality" / "pics",
        count=400,
        has_text_gt=True,
        description="OCR-Quality: multilingual docs with human quality scores",
    ),
    SourceDataset(
        name="sroie",
        image_dir=_E_01 / "forms" / "sroie_icdar2019" / "train" / "images",
        count=300,
        has_text_gt=True,
        description="SROIE train split: camera-captured receipts with text GT",
    ),
]


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class VLMConsensusResult:
    """Result of two-model VLM consensus for a single image."""

    image_id: str
    image_path: str
    primary_label: str | None
    primary_score: float | None
    secondary_label: str | None
    secondary_score: float | None
    tiebreaker_label: str | None = None
    tiebreaker_score: float | None = None
    consensus_mos: float | None = None
    consensus_std: float = 0.8
    agreement: bool = False
    tiebreaker_used: bool = False
    excluded: bool = False
    parse_success: bool = True


@dataclass
class ConsensusTracker:
    """Tracks VLM consensus labeling progress and costs."""

    total_images: int = 0
    labeled: int = 0
    agreements: int = 0
    tiebreakers: int = 0
    exclusions: int = 0
    failures: int = 0
    cost_usd: float = 0.0
    results_by_source: dict[str, int] = field(default_factory=dict)

    # Per-call cost estimates (input ~1500 tokens + output ~5 tokens)
    PRIMARY_COST: float = 0.000105 + 0.0000015  # $0.07/$0.30 per 1M
    SECONDARY_COST: float = 0.0003 + 0.0000044  # $0.20/$0.88 per 1M
    TIEBREAKER_COST: float = 0.003 + 0.00004  # $2/$8 per 1M

    def record_result(
        self, result: VLMConsensusResult, source: str
    ) -> None:
        """Record a consensus result."""
        self.total_images += 1
        # Always pay for primary + secondary
        self.cost_usd += self.PRIMARY_COST + self.SECONDARY_COST

        if result.excluded:
            self.exclusions += 1
        elif result.consensus_mos is not None:
            self.labeled += 1
            self.results_by_source[source] = (
                self.results_by_source.get(source, 0) + 1
            )
            if result.agreement:
                self.agreements += 1
            if result.tiebreaker_used:
                self.tiebreakers += 1
                self.cost_usd += self.TIEBREAKER_COST
        else:
            self.failures += 1

    def summary(self) -> dict:
        """Return summary stats."""
        return {
            "total_images": self.total_images,
            "labeled": self.labeled,
            "agreements": self.agreements,
            "tiebreakers": self.tiebreakers,
            "exclusions": self.exclusions,
            "failures": self.failures,
            "cost_usd": round(self.cost_usd, 4),
            "agreement_rate": (
                self.agreements / self.labeled if self.labeled > 0 else 0.0
            ),
            "results_by_source": dict(self.results_by_source),
        }


# ── API Calls ────────────────────────────────────────────────────────────────


def _parse_vlm_response(response_text: str) -> str | None:
    """Parse VLM response to extract quality level.

    Returns:
        Quality level string or None if parsing failed.
    """
    import re

    text = response_text.strip().lower().rstrip(".")
    for level in QUALITY_LEVEL_MAP:
        if text == level or re.match(rf"^{level}\b", text):
            return level
    return None


def _encode_image(image_path: str | Path) -> tuple[str, str]:
    """Read and base64-encode an image file.

    Returns:
        Tuple of (base64_data, mime_type).
    """
    path = Path(image_path)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
    }.get(suffix, "image/jpeg")
    return b64, mime


def _call_openrouter(
    image_path: str | Path,
    model: str,
    api_key: str,
    prompt: str = QUALITY_PROMPT,
    max_tokens: int = 32,
    temperature: float = 0.0,
    timeout: int = 120,
) -> tuple[str, float]:
    """Make a single API call to OpenRouter.

    Args:
        image_path: Path to image file.
        model: OpenRouter model ID.
        api_key: OpenRouter API key.
        prompt: Text prompt.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (response_text, latency_ms).

    Raises:
        ImportError: If httpx is not installed.
        RuntimeError: If API call fails.
    """
    try:
        import httpx
    except ImportError as exc:
        msg = "httpx required: uv pip install httpx"
        raise ImportError(msg) from exc

    b64_image, mime = _encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    start = time.monotonic()
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    latency_ms = (time.monotonic() - start) * 1000

    if response.status_code != 200:
        msg = f"OpenRouter API error {response.status_code}: {response.text}"
        raise RuntimeError(msg)

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return text, latency_ms


def _call_model_with_retry(
    image_path: str | Path,
    model: str,
    api_key: str,
    retries: int = RETRY_LIMIT,
) -> tuple[str | None, float | None]:
    """Call a VLM model and parse the quality label, with retry on parse failure.

    Returns:
        Tuple of (quality_label, mos_score). Both None if all attempts fail.
    """
    for attempt in range(1 + retries):
        try:
            text, latency = _call_openrouter(image_path, model, api_key)
            label = _parse_vlm_response(text)
            if label is not None:
                return label, QUALITY_LEVEL_MAP[label]
            if attempt < retries:
                logger.debug(
                    "Parse failed for %s (attempt %d): '%s'",
                    model, attempt + 1, text,
                )
        except Exception:
            logger.warning(
                "API call failed for %s (attempt %d)",
                model, attempt + 1, exc_info=True,
            )
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential backoff
    return None, None


# ── Consensus Logic ──────────────────────────────────────────────────────────


def compute_consensus(
    primary_score: float | None,
    secondary_score: float | None,
    tiebreaker_score: float | None = None,
    threshold: float = AGREEMENT_THRESHOLD,
) -> tuple[float | None, float, bool, bool, bool]:
    """Compute consensus MOS from multi-model scores.

    Returns:
        Tuple of (consensus_mos, consensus_std, agreement, tiebreaker_used,
        excluded).
    """
    if primary_score is None and secondary_score is None:
        return None, 0.8, False, False, False

    # Single model fallback
    if primary_score is None:
        return secondary_score, 1.2, False, False, False
    if secondary_score is None:
        return primary_score, 1.2, False, False, False

    disagreement = abs(primary_score - secondary_score)

    if disagreement <= threshold:
        # Agreement: use mean
        consensus = (primary_score + secondary_score) / 2.0
        # Lower disagreement → lower std (base 0.4 + half disagreement)
        std = 0.4 + disagreement / 2.0
        return consensus, std, True, False, False

    # Disagreement: use tiebreaker if available
    if tiebreaker_score is not None:
        import numpy as np

        scores = sorted([primary_score, secondary_score, tiebreaker_score])
        consensus = scores[1]  # median of three
        std = float(np.std([primary_score, secondary_score, tiebreaker_score]))
        return consensus, max(std, 0.6), False, True, False

    # No tiebreaker and high disagreement: exclude
    return None, 0.0, False, False, True


# ── Image Discovery ──────────────────────────────────────────────────────────


def discover_images(source: SourceDataset, seed: int = 42) -> list[Path]:
    """Discover and sample images from a source dataset, respecting split rules.

    Split enforcement:
        - SmartDoc-QA: split_manifest with document-level train/val/test
        - FUNSD+: filename prefix filter (funsd_plus_train_*)
        - SROIE: image_dir points to train/images/ directly
        - RealDAE: glob *_in.* to skip GT pair images
        - Others: no filtering needed

    Args:
        source: Source dataset configuration.
        seed: Random seed for sampling.

    Returns:
        List of image paths, up to source.count.
    """
    if not source.image_dir.exists():
        logger.warning("Source directory not found: %s", source.image_dir)
        return []

    # If a split manifest exists, use it to get train-only image list
    if source.split_manifest and source.split_manifest.exists():
        with open(source.split_manifest) as f:
            manifest = json.load(f)
        train_rel_paths = set(manifest.get("train_images", []))
        images = [
            source.image_dir / rel_path
            for rel_path in train_rel_paths
            if (source.image_dir / rel_path).exists()
        ]
        images = sorted(images)
        logger.info(
            "Source %s: loaded %d train images from split manifest",
            source.name, len(images),
        )
    else:
        # Collect images via glob
        images: list[Path] = []
        for pattern in source.glob_patterns:
            if source.recursive:
                images.extend(source.image_dir.rglob(pattern))
            else:
                images.extend(source.image_dir.glob(pattern))

        # Remove duplicates and sort for determinism
        images = sorted(set(images))

        # Apply split filter (filename prefix)
        if source.split_filter:
            images = [p for p in images if p.name.startswith(source.split_filter)]

        # Apply exclude suffix (e.g., skip _gt pair images)
        if source.exclude_suffix:
            images = [
                p for p in images
                if source.exclude_suffix not in p.stem
            ]

    total_available = len(images)
    if total_available == 0:
        logger.warning("No images found for %s in %s", source.name, source.image_dir)
        return []

    # Sample if we have more than needed
    rng = random.Random(seed)
    if len(images) > source.count:
        images = rng.sample(images, source.count)
    else:
        rng.shuffle(images)

    logger.info(
        "Source %s: %d available (after split filter), selected %d",
        source.name,
        total_available,
        len(images),
    )
    return sorted(images)


# ── Core Labeling ────────────────────────────────────────────────────────────


def label_single_image(
    image_path: str | Path,
    api_key: str,
    primary_model: str = PRIMARY_MODEL,
    secondary_model: str = SECONDARY_MODEL,
    tiebreaker_model: str = TIEBREAKER_MODEL,
    threshold: float = AGREEMENT_THRESHOLD,
) -> VLMConsensusResult:
    """Label a single image using two-model VLM consensus.

    Args:
        image_path: Path to image file.
        api_key: OpenRouter API key.
        primary_model: Primary model ID.
        secondary_model: Secondary model ID.
        tiebreaker_model: Tiebreaker model ID.
        threshold: MOS disagreement threshold for tiebreaker invocation.

    Returns:
        VLMConsensusResult with consensus MOS.
    """
    image_path = Path(image_path)
    image_id = image_path.stem

    # Call primary
    primary_label, primary_score = _call_model_with_retry(
        image_path, primary_model, api_key
    )

    # Call secondary
    secondary_label, secondary_score = _call_model_with_retry(
        image_path, secondary_model, api_key
    )

    # Check if tiebreaker needed
    tiebreaker_label = None
    tiebreaker_score = None
    tiebreaker_used = False

    if (
        primary_score is not None
        and secondary_score is not None
        and abs(primary_score - secondary_score) > threshold
    ):
        tiebreaker_label, tiebreaker_score = _call_model_with_retry(
            image_path, tiebreaker_model, api_key
        )
        tiebreaker_used = tiebreaker_label is not None

    # Compute consensus
    consensus_mos, consensus_std, agreement, tb_used, excluded = compute_consensus(
        primary_score, secondary_score, tiebreaker_score,
        threshold=threshold,
    )

    parse_success = primary_label is not None or secondary_label is not None

    return VLMConsensusResult(
        image_id=image_id,
        image_path=str(image_path),
        primary_label=primary_label,
        primary_score=primary_score,
        secondary_label=secondary_label,
        secondary_score=secondary_score,
        tiebreaker_label=tiebreaker_label,
        tiebreaker_score=tiebreaker_score,
        consensus_mos=consensus_mos,
        consensus_std=consensus_std,
        agreement=agreement,
        tiebreaker_used=tiebreaker_used or tb_used,
        excluded=excluded,
        parse_success=parse_success,
    )


# ── Checkpointing ───────────────────────────────────────────────────────────


def _load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load completed image IDs from checkpoint file.

    Returns:
        Set of image_ids already processed.
    """
    completed: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed.add(entry["image_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Loaded checkpoint: %d images already processed", len(completed))
    return completed


_checkpoint_lock = threading.Lock()


def _append_checkpoint(
    checkpoint_path: Path,
    result: VLMConsensusResult,
    source_name: str,
) -> None:
    """Append a single result to the checkpoint file (thread-safe)."""
    entry = {
        "image_id": result.image_id,
        "source": source_name,
        "image_path": result.image_path,
        "primary": {"label": result.primary_label, "score": result.primary_score},
        "secondary": {"label": result.secondary_label, "score": result.secondary_score},
        "tiebreaker": {
            "label": result.tiebreaker_label,
            "score": result.tiebreaker_score,
            "used": result.tiebreaker_used,
        },
        "consensus_mos": result.consensus_mos,
        "consensus_std": result.consensus_std,
        "agreement": result.agreement,
        "excluded": result.excluded,
        "parse_success": result.parse_success,
    }
    with _checkpoint_lock:
        with open(checkpoint_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Pipeline ─────────────────────────────────────────────────────────────────


def _resolve_doc_split(
    source: SourceDataset,
    image_path: Path,
    split_lookup: dict[str, str],
) -> str:
    """Determine document split for an image."""
    if split_lookup:
        rel = str(image_path.relative_to(source.image_dir))
        return split_lookup.get(rel, "unknown")
    if source.name == "funsd":
        return "test" if "/test/" in str(image_path) else "train"
    if source.name == "funsd_plus":
        return (
            "test" if image_path.name.startswith("funsd_plus_test_")
            else "train"
        )
    return "train"


def _process_single(
    image_path: Path,
    source: SourceDataset,
    api_key: str,
    checkpoint_path: Path,
    output_dir: Path,
) -> VLMConsensusResult:
    """Label one image, checkpoint, and copy to output. Thread-safe."""
    result = label_single_image(image_path, api_key)
    _append_checkpoint(checkpoint_path, result, source.name)

    # Copy image to output if successful
    if result.consensus_mos is not None and not result.excluded:
        import shutil

        images_out = output_dir / "images" / f"stream3_{source.name}"
        images_out.mkdir(parents=True, exist_ok=True)
        out_filename = f"{source.name}_{result.image_id}{image_path.suffix}"
        out_path = images_out / out_filename
        if not out_path.exists():
            shutil.copyfile(image_path, out_path)

    return result


def _rebuild_records_from_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    sources: list[SourceDataset],
    split_lookups: dict[str, dict[str, str]],
    dimension: str = "overall",
    base_seed: int = 40000,
) -> tuple[list[dict], list[dict]]:
    """Rebuild all DeQA training records from the checkpoint file.

    This ensures records exist for ALL successfully labeled images,
    including those from previous runs.

    Returns:
        Tuple of (train_records, all_records).
    """
    source_map = {s.name: s for s in sources}
    train_records: list[dict] = []
    all_records: list[dict] = []
    idx = 0

    with open(checkpoint_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("consensus_mos") is None or entry.get("excluded"):
                idx += 1
                continue

            source_name = entry["source"]
            image_id = entry["image_id"]
            source = source_map.get(source_name)
            if not source:
                idx += 1
                continue

            # Determine split
            image_path = Path(entry["image_path"])
            split_lookup = split_lookups.get(source_name, {})
            doc_split = _resolve_doc_split(source, image_path, split_lookup)

            # Determine output filename and path
            suffix = image_path.suffix
            out_filename = f"{source_name}_{image_id}{suffix}"
            rel_path = (
                f"DIQA-5000_1/images/stream3_{source_name}/{out_filename}"
            )

            record = vlm_scores_to_deqa_record(
                image_id=f"s3_{source_name}_{image_id}",
                image_path=rel_path,
                vlm_mos=entry["consensus_mos"],
                vlm_std=entry["consensus_std"],
                source=source_name,
                weight=VLM_WEIGHT,
                dimension=dimension,
                seed=base_seed + idx,
                vlm_models=[PRIMARY_MODEL, SECONDARY_MODEL],
            )
            record["has_text_gt"] = source.has_text_gt
            record["vlm_agreement"] = entry.get("agreement", False)
            record["tiebreaker_used"] = entry.get("tiebreaker", {}).get(
                "used", False
            )
            record["doc_split"] = doc_split

            all_records.append(record)
            if doc_split == "train":
                train_records.append(record)

            idx += 1

    return train_records, all_records


def generate_stream3(
    sources: list[SourceDataset] | None = None,
    output_dir: str | Path = "",
    api_key: str | None = None,
    dimension: str = "overall",
    base_seed: int = 40000,
    dry_run: bool = False,
    resume: bool = True,
    max_workers: int = 10,
) -> list[dict]:
    """Generate Stream 3 VLM consensus-labeled training samples.

    Full pipeline:
    1. Discover images from configured source datasets (split-aware)
    2. Label each image with two-model VLM consensus (parallel workers)
    3. Checkpoint progress for resume on interruption
    4. Rebuild ALL DeQA training records from the complete checkpoint

    Args:
        sources: List of source datasets. Default: TIER1_SOURCES.
        output_dir: Directory for DIQA-5000_1 output.
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var).
        dimension: Quality dimension to assess.
        base_seed: Base seed for reproducibility.
        dry_run: If True, plan only without API calls.
        resume: If True, resume from checkpoint.
        max_workers: Number of parallel API workers.

    Returns:
        List of DeQA training records (dicts).
    """
    sources = sources or TIER1_SOURCES
    output_dir = Path(output_dir)
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key and not dry_run:
        logger.error("No OPENROUTER_API_KEY set. Cannot make API calls.")
        return []

    # Discover all images first
    all_work: list[tuple[SourceDataset, list[Path]]] = []
    total_planned = 0
    for source in sources:
        images = discover_images(source, seed=base_seed)
        all_work.append((source, images))
        total_planned += len(images)

    if dry_run:
        logger.info("=== Stream 3 Dry Run ===")
        logger.info("Total: %d images from %d sources", total_planned, len(sources))
        for source, images in all_work:
            logger.info(
                "  %-15s %4d images (text_gt=%s)",
                source.name, len(images), source.has_text_gt,
            )
        cost_2model = total_planned * (
            ConsensusTracker.PRIMARY_COST + ConsensusTracker.SECONDARY_COST
        )
        logger.info("  Estimated cost: $%.2f (2-model, excl. tiebreakers)", cost_2model)
        return []

    # Load split manifests
    split_lookups: dict[str, dict[str, str]] = {}
    for source, _ in all_work:
        if source.split_manifest and source.split_manifest.exists():
            with open(source.split_manifest) as f:
                manifest = json.load(f)
            lookup: dict[str, str] = {}
            for split_name in ("train", "val", "test"):
                for rel_path in manifest.get(f"{split_name}_images", []):
                    lookup[rel_path] = split_name
            split_lookups[source.name] = lookup
            logger.info(
                "Loaded split manifest for %s: %d images mapped",
                source.name, len(lookup),
            )

    # Checkpoint setup
    checkpoint_path = output_dir / "vlm_checkpoint.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = _load_checkpoint(checkpoint_path) if resume else set()

    # Build work queue (skip already checkpointed)
    work_items: list[tuple[SourceDataset, Path]] = []
    for source, images in all_work:
        pending = [p for p in images if p.stem not in completed_ids]
        skipped = len(images) - len(pending)
        logger.info(
            "Source %s: %d total, %d checkpointed, %d to process",
            source.name, len(images), skipped, len(pending),
        )
        for img in pending:
            work_items.append((source, img))

    if not work_items:
        logger.info("All images already checkpointed. Rebuilding records.")
    else:
        logger.info(
            "Processing %d images with %d parallel workers",
            len(work_items), max_workers,
        )

        # Parallel labeling
        tracker = ConsensusTracker()
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single,
                    image_path,
                    source,
                    api_key,
                    checkpoint_path,
                    output_dir,
                ): (source, image_path)
                for source, image_path in work_items
            }

            for future in as_completed(futures):
                source, image_path = futures[future]
                try:
                    result = future.result()
                    tracker.record_result(result, source.name)
                    if result.consensus_mos is not None:
                        completed += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                    logger.warning(
                        "Worker exception for %s", image_path.name,
                        exc_info=True,
                    )

                total_done = completed + failed
                if total_done % 100 == 0:
                    logger.info(
                        "Progress: %d/%d done (%d ok, %d failed, cost: $%.4f)",
                        total_done,
                        len(work_items),
                        completed,
                        failed,
                        tracker.cost_usd,
                    )

        logger.info(
            "Labeling complete: %d ok, %d failed. %s",
            completed, failed, json.dumps(tracker.summary()),
        )

    # ── Rebuild ALL records from checkpoint ──────────────────────────────
    logger.info("Rebuilding records from checkpoint...")
    train_records, all_records = _rebuild_records_from_checkpoint(
        checkpoint_path, output_dir, list(sources),
        split_lookups, dimension, base_seed,
    )

    # Save training records (train-split only)
    if train_records:
        records_path = output_dir / "stream3_records.json"
        with open(records_path, "w") as f:
            json.dump(train_records, f, indent=2)
        logger.info("Saved %d training records to %s", len(train_records), records_path)

    # Save ALL VLM-labeled records (all splits) for evaluation
    if all_records:
        all_records_path = output_dir / "stream3_all_vlm_records.json"
        with open(all_records_path, "w") as f:
            json.dump(all_records, f, indent=2)
        from collections import Counter
        split_counts = Counter(r.get("doc_split", "train") for r in all_records)
        logger.info(
            "Saved %d total VLM records to %s (by split: %s)",
            len(all_records), all_records_path, dict(split_counts),
        )

    # Save raw checkpoint as audit JSON
    if checkpoint_path.exists():
        vlm_results_path = output_dir / "vlm_consensus_results.json"
        results_list = []
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    results_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        with open(vlm_results_path, "w") as f:
            json.dump(results_list, f, indent=2)
        logger.info(
            "Saved %d VLM results to %s", len(results_list), vlm_results_path
        )

    return train_records


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def _load_env() -> None:
    """Load .env file from project root into os.environ."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        # Try one more level up (DeQA-Doc/.env)
        env_path = Path(__file__).resolve().parents[4] / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    """CLI entry point for Stream 3 VLM consensus labeling."""
    import argparse

    _load_env()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Stream 3: VLM consensus labeling for DIQA-5000_1"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/e/dataset/training_data/DIQA-5000/DIQA-5000_1",
        help="Output directory for DIQA-5000_1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, no API calls",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore checkpoint)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Run only a specific source (e.g., 'tobacco800')",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel API workers (default: 10)",
    )
    args = parser.parse_args()

    sources = TIER1_SOURCES
    if args.source:
        sources = [s for s in TIER1_SOURCES if s.name == args.source]
        if not sources:
            valid = ", ".join(s.name for s in TIER1_SOURCES)
            parser.error(f"Unknown source '{args.source}'. Valid: {valid}")

    records = generate_stream3(
        sources=sources,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        max_workers=args.workers,
    )

    if records:
        print(f"\nGenerated {len(records)} training records.")


if __name__ == "__main__":
    main()
