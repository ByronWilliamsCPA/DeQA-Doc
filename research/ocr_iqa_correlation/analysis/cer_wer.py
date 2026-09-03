"""CER, WER, and FCA computation using jiwer.

Computes Character Error Rate, Word Error Rate, and Flexible Character
Accuracy between OCR output and ground truth text. Includes text
normalization for fair comparison.

FCA (Flexible Character Accuracy) was developed by the OCR-D project to
mitigate CER's sensitivity to reading-order errors. It splits text into
lines, finds optimal line-level alignment, and computes average CER across
aligned pairs — making it robust to layout-level divergence where OCR
engines segment text blocks differently from the ground truth.

Reference:
    Clausner, Pletschacher, Antonacopoulos (2020). "Flexible Character
    Accuracy — an evaluation metric for OCR." ICPR 2020.
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


def _split_into_segments(
    text: str, *, target_segment_chars: int = 80
) -> list[str]:
    """Split text into segments for FCA alignment.

    If the text contains multiple non-empty lines, uses natural line breaks.
    If the text is a single line (common for concatenated form entities),
    splits on word boundaries into segments of approximately
    target_segment_chars characters each.

    Args:
        text: Input text (may or may not contain newlines).
        target_segment_chars: Target character count per segment when
            splitting single-line text. Default 80 (typical OCR line width).

    Returns:
        List of non-empty normalized segments.
    """
    # Try natural line splitting first
    lines = []
    for line in text.splitlines():
        normalized = normalize_text(line)
        if normalized:
            lines.append(normalized)

    if len(lines) > 1:
        return lines

    # Single line or no lines — split on word boundaries
    full_text = normalize_text(text)
    if not full_text:
        return []

    words = full_text.split()
    if not words:
        return []

    segments = []
    current_segment: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word) + (1 if current_segment else 0)
        if current_len + word_len > target_segment_chars and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = [word]
            current_len = len(word)
        else:
            current_segment.append(word)
            current_len += word_len

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments


def _line_cer(ref_line: str, hyp_line: str) -> float:
    """Compute CER between two pre-normalized lines.

    Args:
        ref_line: Normalized reference line.
        hyp_line: Normalized hypothesis line.

    Returns:
        CER as a float.
    """
    import jiwer

    if not ref_line and not hyp_line:
        return 0.0
    if not ref_line or not hyp_line:
        return 1.0
    return jiwer.cer(ref_line, hyp_line)


def compute_fca(reference: str, hypothesis: str) -> float:
    """Compute Flexible Character Accuracy (FCA).

    Splits reference and hypothesis into lines, finds the best greedy
    alignment by minimizing per-line CER, then returns the mean CER
    across aligned pairs. Unmatched lines contribute CER = 1.0.

    This is robust to reading-order differences where OCR engines may
    segment text blocks differently from the ground truth (e.g., two
    columns read as one block, or vice versa).

    The metric is returned as an error rate (like CER), not an accuracy,
    despite the name "Flexible Character Accuracy" — this keeps it
    consistent with CER/WER in this module. Lower is better.

    Args:
        reference: Ground truth text (may contain newlines).
        hypothesis: OCR output text (may contain newlines).

    Returns:
        FCA error rate as a float in [0, 1]. Returns 0.0 if both are
        empty; 1.0 if one is empty.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm or not hyp_norm:
        return 1.0

    ref_lines = _split_into_segments(reference)
    hyp_lines = _split_into_segments(hypothesis)

    if not ref_lines and not hyp_lines:
        return 0.0
    if not ref_lines or not hyp_lines:
        return 1.0

    # Greedy alignment: for each ref line, find the best matching hyp line
    # This is O(n*m) but n and m are small (typically <50 lines per page)
    used_hyp: set[int] = set()
    matched_cers: list[float] = []

    for ref_line in ref_lines:
        best_cer = 1.0
        best_idx = -1

        for j, hyp_line in enumerate(hyp_lines):
            if j in used_hyp:
                continue
            cer = _line_cer(ref_line, hyp_line)
            if cer < best_cer:
                best_cer = cer
                best_idx = j

        if best_idx >= 0:
            used_hyp.add(best_idx)
        matched_cers.append(best_cer)

    # Unmatched hypothesis lines contribute CER = 1.0 each,
    # weighted by the ratio of unmatched hyp chars to total ref chars
    unmatched_hyp_count = len(hyp_lines) - len(used_hyp)
    for _ in range(unmatched_hyp_count):
        matched_cers.append(1.0)

    return sum(matched_cers) / len(matched_cers) if matched_cers else 0.0


def compute_metrics(
    reference: str, hypothesis: str, *, include_fca: bool = False
) -> dict[str, float]:
    """Compute CER, WER, and optionally FCA for a single pair.

    Args:
        reference: Ground truth text.
        hypothesis: OCR output text.
        include_fca: If True, also compute FCA (slightly slower).

    Returns:
        Dict with 'cer', 'wer', and optionally 'fca' keys.
    """
    result = {
        "cer": compute_cer(reference, hypothesis),
        "wer": compute_wer(reference, hypothesis),
    }
    if include_fca:
        result["fca"] = compute_fca(reference, hypothesis)
    return result
