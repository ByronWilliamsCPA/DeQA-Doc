"""OCR engine protocol and result types.

Defines the interface that all OCR engines must implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OCRResult:
    """Result from an OCR engine run on a single image.

    Attributes:
        text: Extracted text content.
        time_ms: Processing time in milliseconds.
        engine_name: Name of the OCR engine that produced this result.
        error: Error message if OCR failed, None otherwise.
    """

    text: str
    time_ms: float
    engine_name: str
    error: str | None = None


@runtime_checkable
class OCREngine(Protocol):
    """Protocol for OCR engines."""

    @property
    def name(self) -> str:
        """Engine identifier."""
        ...

    def recognize(self, image_path: str) -> OCRResult:
        """Run OCR on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        ...
