"""CER and WER computation using jiwer.

Computes Character Error Rate and Word Error Rate between OCR output
and ground truth text. Includes text normalization for fair comparison.
"""

from __future__ import annotations

import logging
import unicodedata

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for CER/WER comparison.

    Applies:
    - Unicode NFC normalization
    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Raw text string.

    Returns:
        Normalized text.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate.

    Args:
        reference: Ground truth text.
        hypothesis: OCR output text.

    Returns:
        CER as a float in [0, inf). Returns 1.0 if reference is empty
        but hypothesis is not; returns 0.0 if both are empty.
    """
    import jiwer

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    if not hyp:
        return 1.0

    return jiwer.cer(ref, hyp)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate.

    Args:
        reference: Ground truth text.
        hypothesis: OCR output text.

    Returns:
        WER as a float in [0, inf). Returns 1.0 if reference is empty
        but hypothesis is not; returns 0.0 if both are empty.
    """
    import jiwer

    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    if not hyp:
        return 1.0

    return jiwer.wer(ref, hyp)


def compute_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute both CER and WER for a single pair.

    Args:
        reference: Ground truth text.
        hypothesis: OCR output text.

    Returns:
        Dict with 'cer' and 'wer' keys.
    """
    return {
        "cer": compute_cer(reference, hypothesis),
        "wer": compute_wer(reference, hypothesis),
    }
