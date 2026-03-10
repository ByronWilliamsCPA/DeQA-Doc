"""PaddleOCR PP-OCRv5 engine.

Uses PaddlePaddle's PP-OCRv5 model (the traditional OCR engine, not
PaddleOCR-VL). Requires paddlepaddle and paddleocr packages.

Install:
    pip install paddlepaddle paddleocr
"""

from __future__ import annotations

import logging
import time

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """OCR engine using PaddleOCR PP-OCRv5.

    Args:
        use_server_model: Use the larger server models for higher accuracy.
            Defaults to False (uses mobile models).
    """

    def __init__(self, *, use_server_model: bool = False) -> None:
        self._use_server_model = use_server_model
        self._ocr = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "paddleocr"

    def _get_ocr(self):
        """Lazily initialize PaddleOCR."""
        if self._ocr is not None:
            return self._ocr

        import os

        # Disable model source connectivity check (slow on some networks)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        from paddleocr import PaddleOCR

        kwargs: dict = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }

        if self._use_server_model:
            kwargs["text_detection_model_name"] = "PP-OCRv5_server_det"
            kwargs["text_recognition_model_name"] = "PP-OCRv5_server_rec"
        else:
            kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"

        self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def recognize(self, image_path: str) -> OCRResult:
        """Run PP-OCRv5 on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            ocr = self._get_ocr()
            result = ocr.predict(image_path)

            # Extract recognized text from result structure.
            # PaddleOCR 3.x returns a generator of dict-like OCRResult
            # objects with a 'rec_texts' key containing list of strings.
            text_parts: list[str] = []
            for page_result in result:
                rec_texts = page_result.get("rec_texts", [])
                text_parts.extend(rec_texts)

            text = "\n".join(text_parts)
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("PaddleOCR failed on %s", image_path)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
