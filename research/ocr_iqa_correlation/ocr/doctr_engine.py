"""docTR OCR engine.

Uses mindee/doctr for document text recognition with a two-stage
pipeline (text detection + recognition). PyTorch backend.

Install:
    pip install "python-doctr[torch]"
"""

from __future__ import annotations

import logging
import time

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class DocTROCREngine:
    """OCR engine using docTR (mindee/doctr).

    Args:
        det_arch: Detection model architecture.
        reco_arch: Recognition model architecture.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
    ) -> None:
        self._det_arch = det_arch
        self._reco_arch = reco_arch
        self._model = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "doctr"

    def _get_model(self):
        """Lazily initialize the docTR predictor."""
        if self._model is not None:
            return self._model

        from doctr.models import ocr_predictor

        self._model = ocr_predictor(
            det_arch=self._det_arch,
            reco_arch=self._reco_arch,
            pretrained=True,
        )
        return self._model

    def recognize(self, image_path: str) -> OCRResult:
        """Run docTR on a single image.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            from doctr.io import DocumentFile

            model = self._get_model()
            doc = DocumentFile.from_images([image_path])
            result = model(doc)
            text = result.render()
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("docTR failed on %s", image_path)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
