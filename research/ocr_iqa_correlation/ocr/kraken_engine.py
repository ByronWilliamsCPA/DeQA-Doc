"""Kraken OCR engine.

Uses kraken for document text recognition with baseline segmentation.
Specialized for historical and non-Latin documents.

Install:
    pip install kraken
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from research.ocr_iqa_correlation.ocr.base import OCRResult

logger = logging.getLogger(__name__)

# Default recognition model — community English printed text model.
# Downloaded on first use via kraken.repo.get_model().
DEFAULT_REC_MODEL = "en_best.mlmodel"


class KrakenOCREngine:
    """OCR engine using Kraken.

    Args:
        rec_model: Path or name of the recognition model. If a name is
            given (e.g. 'en_best.mlmodel'), it will be downloaded from
            the Kraken model repository on first use.
    """

    def __init__(self, rec_model: str = DEFAULT_REC_MODEL) -> None:
        self._rec_model_spec = rec_model
        self._rec_model = None
        self._seg_model = None

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "kraken"

    def _load_models(self) -> None:
        """Lazily load segmentation and recognition models."""
        if self._rec_model is not None:
            return

        from kraken.lib import models as kraken_models

        rec_path = Path(self._rec_model_spec)
        if not rec_path.exists():
            # Search in Kraken's default model directory (htrmopo)
            import glob

            model_dirs = glob.glob(
                str(Path.home() / ".local/share/htrmopo/*/")
            )
            for model_dir in model_dirs:
                candidate = Path(model_dir) / self._rec_model_spec
                if candidate.exists():
                    rec_path = candidate
                    break
            else:
                # Download via CLI subprocess as fallback
                import subprocess
                import sys

                logger.info(
                    "Downloading Kraken model: %s", self._rec_model_spec
                )
                subprocess.run(
                    [
                        sys.executable.replace(
                            "/bin/python", "/bin/kraken"
                        ),
                        "get",
                        "10.5281/zenodo.2577813",
                    ],
                    check=True,
                    capture_output=True,
                )
                # Re-search after download
                for model_dir in glob.glob(
                    str(Path.home() / ".local/share/htrmopo/*/")
                ):
                    candidate = Path(model_dir) / self._rec_model_spec
                    if candidate.exists():
                        rec_path = candidate
                        break

        self._rec_model = kraken_models.load_any(str(rec_path))
        logger.info("Kraken recognition model loaded: %s", rec_path)

    def recognize(self, image_path: str) -> OCRResult:
        """Run Kraken on a single image.

        Uses baseline segmentation then recognition.

        Args:
            image_path: Path to the image file.

        Returns:
            OCRResult with extracted text and timing.
        """
        start = time.monotonic()

        try:
            from PIL import Image

            from kraken import blla, rpred

            self._load_models()
            im = Image.open(image_path)

            # Segment the page using built-in baseline segmentation
            baseline_seg = blla.segment(im)

            # Recognize text
            text_lines: list[str] = []
            for record in rpred.rpred(self._rec_model, im, baseline_seg):
                text_lines.append(record.prediction)

            text = "\n".join(text_lines)
            elapsed_ms = (time.monotonic() - start) * 1000

            return OCRResult(
                text=text,
                time_ms=elapsed_ms,
                engine_name=self.name,
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("Kraken failed on %s", image_path)
            return OCRResult(
                text="",
                time_ms=elapsed_ms,
                engine_name=self.name,
                error=str(e),
            )
