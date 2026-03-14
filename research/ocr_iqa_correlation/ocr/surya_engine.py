"""Surya OCR engine.

Uses VikParuchuri/surya for document text recognition. Optimized for
documents; requires GPU for reasonable performance on large batches.

Install:
    pip install surya-ocr
"""

from __future__ import annotations

import logging
import time

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class SuryaOCREngine:
    """OCR engine using Surya.

    Best run on GPU — default batch size uses ~13GB VRAM.
    For CPU or low-VRAM GPUs, set RECOGNITION_BATCH_SIZE env var.
    """

    def __init__(self) -> None:
        self._det_predictor = None
        self._rec_predictor = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "surya"

    def _load_models(self) -> None:
        """Lazily load detection and recognition predictors."""
        if self._rec_predictor is not None:
            return

        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        self._det_predictor = DetectionPredictor()
        self._rec_predictor = RecognitionPredictor()
        logger.info("Surya models loaded")

    def recognize(self, image_path: str) -> OCRResult:
        """Run Surya on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            from PIL import Image

            self._load_models()
            im = Image.open(image_path).convert("RGB")

            predictions = self._rec_predictor(
                [im], det_predictor=self._det_predictor
            )

            text_lines: list[str] = []
            for page in predictions:
                for line in page.text_lines:
                    text_lines.append(line.text)

            text = "\n".join(text_lines)
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("Surya failed on %s", image_path)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
