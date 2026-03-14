"""Google Cloud Vision OCR engine.

Uses the DOCUMENT_TEXT_DETECTION endpoint for high-accuracy document OCR.
Requires GOOGLE_APPLICATION_CREDENTIALS environment variable or default
application credentials.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class GoogleVisionOCREngine:
    """OCR engine using Google Cloud Vision API."""

    def __init__(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "gcloud_vision"

    def _get_client(self):
        """Lazily initialize the Vision API client."""
        if self._client is not None:
            return self._client

        from google.cloud import vision

        self._client = vision.ImageAnnotatorClient()
        return self._client

    def recognize(self, image_path: str) -> OCRResult:
        """Run document OCR via Google Cloud Vision.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        from google.cloud import vision

        start = time.monotonic()

        try:
            client = self._get_client()

            with open(image_path, "rb") as f:
                content = f.read()

            image = vision.Image(content=content)
            response = client.document_text_detection(image=image)

            if response.error.message:
                elapsed_ms = (time.monotonic() - start) * 1000
                return OCRResult(
                    text="",
                    time_ms=elapsed_ms,
                    engine_name=self.name,
                    error=response.error.message,
                )

            text = ""
            if response.full_text_annotation:
                text = response.full_text_annotation.text

            elapsed_ms = (time.monotonic() - start) * 1000
            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("Google Vision failed on %s: %s", image_path, e)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
