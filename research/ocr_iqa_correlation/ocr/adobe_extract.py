"""Adobe PDF Extract API OCR engine.

Uses the Adobe PDF Services API to extract text from document images.
Images are first converted to single-page PDFs, then submitted to the
Extract API which returns structured JSON with text content.

Requires Adobe PDF Services credentials:
    - ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET in .env
    - Or pdfservices-api-credentials.json in repo root

Install: pip install pdfservices-sdk Pillow
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)


class AdobeExtractOCREngine:
    """OCR engine using Adobe PDF Extract API.

    Converts images to PDF, submits to Adobe Extract API, and
    returns the extracted text content.
    """

    def __init__(self) -> None:
        self._credentials = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "adobe_extract"

    def _get_credentials(self):
        """Lazily initialize Adobe credentials."""
        if self._credentials is not None:
            return self._credentials

        import os

        from adobe.pdfservices.operation.auth.service_principal_credentials import (
            ServicePrincipalCredentials,
        )

        client_id = os.environ.get("ADOBE_CLIENT_ID", "")
        client_secret = os.environ.get("ADOBE_CLIENT_SECRET", "")

        if not client_id or not client_secret:
            msg = (
                "Adobe credentials not configured. Set ADOBE_CLIENT_ID and "
                "ADOBE_CLIENT_SECRET in .env"
            )
            raise RuntimeError(msg)

        self._credentials = ServicePrincipalCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        return self._credentials

    @staticmethod
    def _image_to_pdf(image_path: str) -> bytes:
        """Convert an image to a single-page PDF in memory.

        Args:
            image_path: Path to the image file.

        Returns:
            PDF file content as bytes.
        """
        from io import BytesIO

        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        pdf_buffer = BytesIO()
        img.save(pdf_buffer, format="PDF", resolution=150.0)
        return pdf_buffer.getvalue()

    def recognize(self, image_path: str) -> OCRResult:
        """Run OCR on a single image via Adobe PDF Extract API.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            from io import BytesIO

            from adobe.pdfservices.operation.config.client_config import ClientConfig
            from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
            from adobe.pdfservices.operation.io.stream_asset import StreamAsset
            from adobe.pdfservices.operation.pdf_services import PDFServices
            from adobe.pdfservices.operation.pdf_services_media_type import (
                PDFServicesMediaType,
            )
            from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import (
                ExtractPDFJob,
            )
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
                ExtractElementType,
            )
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
                ExtractPDFParams,
            )
            from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
                ExtractPDFResult,
            )

            credentials = self._get_credentials()
            pdf_services = PDFServices(credentials=credentials)

            # Convert image to PDF
            pdf_bytes = self._image_to_pdf(image_path)
            input_asset = pdf_services.upload(
                input_stream=BytesIO(pdf_bytes),
                mime_type=PDFServicesMediaType.PDF,
            )

            # Configure extraction
            extract_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT],
            )

            # Submit job
            extract_job = ExtractPDFJob(
                input_asset=input_asset,
                extract_pdf_params=extract_params,
            )

            location = pdf_services.submit(extract_job)
            response = pdf_services.get_job_result(
                location, ExtractPDFResult
            )

            # Extract text from structured JSON result
            import json
            import zipfile

            result_asset: CloudAsset = response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)
            result_bytes = stream_asset.get_input_stream()

            # Adobe returns a ZIP containing structuredData.json
            zip_buffer = BytesIO(result_bytes)
            text_parts = []
            with zipfile.ZipFile(zip_buffer) as zf:
                if "structuredData.json" in zf.namelist():
                    structured = json.loads(zf.read("structuredData.json"))
                    for element in structured.get("elements", []):
                        if "Text" in element:
                            text_parts.append(element["Text"])

            text = "\n".join(text_parts)
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("Adobe Extract failed on %s: %s", image_path, e)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
