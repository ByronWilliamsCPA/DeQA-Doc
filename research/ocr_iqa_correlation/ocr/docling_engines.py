"""Docling-based OCR engines: Tesseract, RapidOCR, EasyOCR.

Each engine uses Docling's DocumentConverter with different OCR backend
configurations. All share the same document processing pipeline but
differ in the underlying OCR library used for text extraction.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class DoclingOCREngine:
    """OCR engine using Docling with a configurable backend.

    Args:
        engine_name: One of 'tesseract', 'rapidocr', 'easyocr'.
    """

    def __init__(self, engine_name: str) -> None:
        self._engine_name = engine_name
        self._converter = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return self._engine_name

    def _get_converter(self):
        """Lazily initialize the Docling converter with the chosen backend."""
        if self._converter is not None:
            return self._converter

        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            OcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Configure OCR backend
        if self._engine_name == "tesseract":
            from docling.datamodel.pipeline_options import TesseractOcrOptions
            ocr_options = TesseractOcrOptions()
        elif self._engine_name == "easyocr":
            from docling.datamodel.pipeline_options import EasyOcrOptions
            ocr_options = EasyOcrOptions()
        elif self._engine_name == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions
            ocr_options = RapidOcrOptions()
        else:
            msg = f"Unknown Docling OCR engine: {self._engine_name}"
            raise ValueError(msg)

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_options,
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_options=pipeline_options
                ),
            }
        )
        return self._converter

    def recognize(self, image_path: str) -> OCRResult:
        """Run OCR on a single image via Docling.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            converter = self._get_converter()
            result = converter.convert(image_path)
            text = result.document.export_to_text()
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self._engine_name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("Docling %s failed on %s: %s", self._engine_name, image_path, e)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self._engine_name,
                error=str(e),
            )


def create_tesseract_engine() -> DoclingOCREngine:
    """Create a Tesseract OCR engine via Docling."""
    return DoclingOCREngine("tesseract")


def create_rapidocr_engine() -> DoclingOCREngine:
    """Create a RapidOCR engine via Docling."""
    return DoclingOCREngine("rapidocr")


def create_easyocr_engine() -> DoclingOCREngine:
    """Create an EasyOCR engine via Docling."""
    return DoclingOCREngine("easyocr")
