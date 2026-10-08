"""Open-source OCR / document-understanding processing engine.

Pipeline per page:

1. use the embedded text layer when present (exact text, zero cost);
2. otherwise render the page and run the configured OCR backend
   (EasyOCR / Tesseract — loaded lazily and cached across requests);
3. detect table structure dynamically from ruling lines and/or
   consistent column gaps;
4. emit the same normalized output as the OpenAI-compatible processor.

The OCR model is fully configurable through ``OCR_MODEL`` /
``OCR_LANGUAGES`` / ``DEVICE``.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

from PIL import Image

from ...config import Settings
from ...debug import DebugWriter
from ...exceptions import OcrError
from ...pdf import PdfDocument, WordBox
from ...schemas import ExtractionResult, PageData
from .. import OPENSOURCE_MODE, register_engine
from ..base import ExtractionEngine
from .ocr import OcrBackend, create_ocr_backend
from .structure import detect_page_tables

logger = logging.getLogger(__name__)

#: pages with at least this many embedded words use the text layer directly
MIN_NATIVE_WORDS = 5

@register_engine
class OpenSourceEngine(ExtractionEngine):
    """Local pipeline: PDF text/OCR + dynamic table structure detection."""

    name = OPENSOURCE_MODE
    description = "Open-Source Model"

    def __init__(self, settings: Settings, *, ocr_backend: Optional[OcrBackend] = None) -> None:
        super().__init__(settings)
        self._ocr_backend = ocr_backend
        self._backend_resolved = ocr_backend is not None

    @property
    def model_name(self) -> Optional[str]:
        return self.settings.ocr_model

    def _get_ocr(self) -> Optional[OcrBackend]:
        """Create the configured OCR backend once and cache it."""
        if not self._backend_resolved:
            self._ocr_backend = create_ocr_backend(self.settings)
            self._backend_resolved = True
            logger.info(
                "OCR backend resolved: %s",
                type(self._ocr_backend).__name__ if self._ocr_backend else "disabled (native)",
            )
        return self._ocr_backend

    # ------------------------------------------------------------------
    def extract(
        self, pdf: PdfDocument, *, debug: Optional[DebugWriter] = None
    ) -> ExtractionResult:
        result = ExtractionResult(
            source_filename=pdf.source_name,
            page_count=pdf.page_count,
            processing_mode=self.name,
            model_name=self.settings.ocr_model,
        )
        for page_number in range(1, pdf.page_count + 1):
            page = self._process_page(pdf, page_number, result, debug)
            result.pages.append(page)
        logger.info(
            "Open-source extraction finished: %d page(s), %d table(s), %d warning(s)",
            len(result.pages),
            result.table_count,
            len(result.warnings),
        )
        return result

    # ------------------------------------------------------------------
    def _process_page(
        self,
        pdf: PdfDocument,
        page_number: int,
        result: ExtractionResult,
        debug: Optional[DebugWriter],
    ) -> PageData:
        native_words = pdf.native_words(page_number)
        grid_scale: Tuple[float, float] = (1.0, 1.0)
        rendered: Optional[bytes] = None

        if len(native_words) >= MIN_NATIVE_WORDS:
            words = native_words
            space_w, space_h = pdf.page_size(page_number)
        else:
            outcome = self._page_without_text(
                pdf, page_number, native_words, result, debug
            )
            if outcome is None:
                return PageData(page_number=page_number, tables=[])
            words, space_w, space_h, grid_scale, rendered = outcome

        h_segments, v_segments = pdf.ruling_segments(page_number)
        tables = detect_page_tables(
            page_number=page_number,
            words=words,
            h_segments=h_segments,
            v_segments=v_segments,
            space_w=space_w,
            space_h=space_h,
            grid_scale=grid_scale,
        )

        if debug is not None:
            if rendered is None:
                rendered = pdf.render_png(page_number, self.settings.ocr_dpi)
            debug.save_page_image(page_number, rendered)
            debug.save_json(
                f"tables_page_{page_number:03d}.json",
                [table.model_dump(mode="json") for table in tables],
            )

        logger.info("page %d: %d word(s), %d table(s)", page_number, len(words), len(tables))
        return PageData(page_number=page_number, tables=tables)

    def _page_without_text(
        self,
        pdf: PdfDocument,
        page_number: int,
        native_words: List[WordBox],
        result: ExtractionResult,
        debug: Optional[DebugWriter],
    ) -> Optional[Tuple[List[WordBox], float, float, Tuple[float, float], Optional[bytes]]]:
        """OCR path for pages without (enough) embedded text."""
        backend = self._get_ocr()
        if backend is None:
            if native_words:
                warning = (
                    f"Page {page_number}: OCR is disabled (OCR_MODEL=native); "
                    "using the limited embedded text only."
                )
                logger.warning(warning)
                result.warnings.append(warning)
                space_w, space_h = pdf.page_size(page_number)
                return native_words, space_w, space_h, (1.0, 1.0), None
            warning = (
                f"Page {page_number} has no text layer and OCR is disabled "
                "(OCR_MODEL=native); page skipped."
            )
            logger.warning(warning)
            result.warnings.append(warning)
            return None

        try:
            return self._ocr_page(pdf, page_number, backend, debug)
        except OcrError as exc:
            # one broken page must not kill the whole conversion
            logger.warning("OCR failed on page %d: %s", page_number, exc.message)
            result.warnings.append(f"Page {page_number}: {exc.message}")
            if native_words:
                space_w, space_h = pdf.page_size(page_number)
                return native_words, space_w, space_h, (1.0, 1.0), None
            return None

    def _ocr_page(
        self,
        pdf: PdfDocument,
        page_number: int,
        backend: OcrBackend,
        debug: Optional[DebugWriter],
    ) -> Tuple[List[WordBox], float, float, Tuple[float, float], Optional[bytes]]:
        png = pdf.render_png(page_number, self.settings.ocr_dpi)
        image = Image.open(io.BytesIO(png))
        words = backend.run(image, page_number)

        if debug is not None:
            debug.save_page_image(page_number, png)
            debug.save_json(
                f"ocr_page_{page_number:03d}.json",
                [
                    {
                        "text": w.text,
                        "bbox": [w.x0, w.y0, w.x1, w.y1],
                        "confidence": w.confidence,
                    }
                    for w in words
                ],
            )

        page_w, page_h = pdf.page_size(page_number)
        scale = (image.width / page_w, image.height / page_h)
        return words, float(image.width), float(image.height), scale, png


__all__ = ["OpenSourceEngine", "MIN_NATIVE_WORDS"]



