"""OCR service: PDF/image input -> text + bounding boxes + confidence.

This is the single entry point for everything OCR-related:

* PDF/image loading (delegates to :mod:`pdf2xlsx.pdf`)
* native text-layer extraction (exact, zero cost)
* OCR backends: ``easyocr`` / ``tesseract`` / ``trocr`` / ``paddleocr`` /
  ``native`` (see :mod:`pdf2xlsx.engines.opensource.ocr`)
* uniform output: :class:`OcrWord` with text, bbox, confidence, page_number

The table-structure detectors (``layout.py`` / ``structure.py``) and the
pipeline consume only this module — never backend imports directly.
This keeps the future TrOCR + Gemma-2B hybrid pluggable: it will be
another post-processor inside this service (see ``ocr_postprocessor``).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image

from .config import Settings
from .debug import DebugWriter
from .engines.opensource.ocr import (
    SUPPORTED_OCR_MODELS,
    OcrBackend,
    create_ocr_backend,
    resolve_device,
)
from .pdf import PdfDocument, PdfSource, WordBox, load_pdf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundingBoxPixels:
    """Pixel-space bounding box (image coordinates)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class OcrWord:
    """One OCR/native word with geometry, confidence and page number."""

    text: str
    bbox: BoundingBoxPixels
    confidence: Optional[float]
    page_number: int

    @property
    def x0(self) -> float:  # convenience for layout code
        return self.bbox.x0

    @property
    def y0(self) -> float:
        return self.bbox.y0

    @property
    def x1(self) -> float:
        return self.bbox.x1

    @property
    def y1(self) -> float:
        return self.bbox.y1

    def to_word_box(self) -> WordBox:
        return WordBox(
            text=self.text,
            x0=self.bbox.x0,
            y0=self.bbox.y0,
            x1=self.bbox.x1,
            y1=self.bbox.y1,
            confidence=self.confidence,
        )

    @staticmethod
    def from_word_box(word: WordBox, page_number: int) -> "OcrWord":
        return OcrWord(
            text=word.text,
            bbox=BoundingBoxPixels(x0=word.x0, y0=word.y0, x1=word.x1, y1=word.y1),
            confidence=word.confidence,
            page_number=page_number,
        )


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------


def load_pdf_input(source: PdfSource, *, source_name: str | None = None) -> PdfDocument:
    """Load a PDF from bytes or a filesystem path."""
    return load_pdf(source, source_name=source_name)


def load_image_input(path: str | Path, *, dpi: int = 200) -> Tuple[Image.Image, str]:
    """Load a single image file (PNG/JPG/...) as a one-page document image."""
    p = Path(path)
    image = Image.open(p).convert("RGB")
    return image, p.name


def render_page_image(pdf: PdfDocument, page_number: int, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes (for OCR / vision models)."""
    return pdf.render_png(page_number, dpi)


def page_image_size(png_bytes: bytes) -> Tuple[int, int]:
    with Image.open(io.BytesIO(png_bytes)) as img:
        return img.width, img.height


# ---------------------------------------------------------------------------
# backend management
# ---------------------------------------------------------------------------


def get_ocr_backend(settings: Settings) -> Optional[OcrBackend]:
    """Create (or resolve to None when disabled) the configured OCR backend.

    When ``OCR_MODEL=trocr`` + ``OCR_POSTPROCESSOR=gemma`` (or the
    ``OCR_MODEL=trocr+gemma`` shorthand), the returned backend is the
    TrOCR + Gemma-2B hybrid: TrOCR proposes line texts, Gemma 2B corrects
    them before word-box alignment (see
    :mod:`pdf2xlsx.engines.opensource.gemma_postprocessor`).
    """
    return create_ocr_backend(settings)


def supported_backends() -> Tuple[str, ...]:
    return tuple(SUPPORTED_OCR_MODELS)


def correct_ocr_lines(
    lines: Sequence[str],
    settings: Settings,
    *,
    corrector: Any | None = None,
) -> List[str]:
    """Correct raw TrOCR line proposals with the configured post-processor.

    Pure service-level helper (no image I/O): useful for retrying or
    unit-testing the Gemma 2B stage in isolation. Returns the input
    unchanged when ``OCR_POSTPROCESSOR=none``.
    """
    from .engines.opensource.gemma_postprocessor import NoOpCorrector, create_corrector

    active = corrector
    if active is None:
        active = create_corrector(
            getattr(settings, "ocr_postprocessor", "none") or "none",
            model_id=getattr(settings, "gemma_model", "google/gemma-2-2b-it"),
            device=resolve_device(settings.device),
            max_new_tokens=int(getattr(settings, "gemma_max_new_tokens", 256)),
        )
    if isinstance(active, NoOpCorrector):
        return list(lines)
    try:
        fixed = active.correct_lines(list(lines))
    except Exception:  # pragma: no cover - corrector already fallbacks internally
        logger.warning("line correction failed; keeping raw OCR text", exc_info=True)
        return list(lines)
    if len(fixed) != len(lines):
        logger.warning("corrector changed line count; keeping raw OCR text")
        return list(lines)
    return [f if f and f.strip() else o for f, o in zip(fixed, lines)]


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def extract_native_words(pdf: PdfDocument, page_number: int) -> List[OcrWord]:
    """Words from the embedded PDF text layer with page numbers attached."""
    return [OcrWord.from_word_box(w, page_number) for w in pdf.native_words(page_number)]


def run_ocr_on_image(
    image: Image.Image,
    page_number: int,
    backend: OcrBackend,
    *,
    debug: Optional[DebugWriter] = None,
) -> List[OcrWord]:
    """Run an OCR backend on a PIL image; attach page numbers."""
    words = backend.run(image, page_number)
    result = [OcrWord.from_word_box(w, page_number) for w in words]
    if debug is not None:
        debug.save_json(
            f"ocr_page_{page_number:03d}.json",
            [
                {
                    "text": w.text,
                    "bbox": [w.x0, w.y0, w.x1, w.y1],
                    "confidence": w.confidence,
                    "page_number": w.page_number,
                }
                for w in result
            ],
        )
    return result


def extract_page_words(
    pdf: PdfDocument,
    page_number: int,
    settings: Settings,
    *,
    backend: Optional[OcrBackend] = None,
    min_native_words: int = 5,
    debug: Optional[DebugWriter] = None,
) -> Tuple[List[WordBox], Tuple[float, float], Tuple[float, float]]:
    """Best-effort words for one page.

    Returns ``(words, (space_w, space_h), grid_scale)`` where the space is
    the coordinate system of ``words`` (PDF points for native text, pixels
    for OCR) and ``grid_scale`` maps ruling-line coordinates into it.

    * native text layer when it has >= ``min_native_words`` words;
    * otherwise the configured OCR backend (PaddleOCR included);
    * empty list when OCR is disabled and no text layer exists.
    """
    from .exceptions import OcrError

    native = pdf.native_words(page_number)
    if len(native) >= min_native_words:
        w, h = pdf.page_size(page_number)
        return native, (float(w), float(h)), (1.0, 1.0)

    active = backend if backend is not None else create_ocr_backend(settings)
    if active is None:
        if native:
            logger.warning(
                "page %d: OCR disabled (OCR_MODEL=native); using embedded text only",
                page_number,
            )
            w, h = pdf.page_size(page_number)
            return native, (float(w), float(h)), (1.0, 1.0)
        return [], pdf.page_size(page_number), (1.0, 1.0)

    try:
        png = pdf.render_png(page_number, settings.ocr_dpi)
    except Exception as exc:
        raise OcrError(f"Could not render page {page_number} for OCR.") from exc
    image = Image.open(io.BytesIO(png))
    ocr_words = [w.to_word_box() if isinstance(w, OcrWord) else w for w in active.run(image, page_number)]
    # WordBox list expected downstream; keep OcrWord conversion available.
    boxes: List[WordBox] = list(ocr_words)  # type: ignore[arg-type]
    if debug is not None:
        debug.save_page_image(page_number, png)
    page_w, page_h = pdf.page_size(page_number)
    scale = (image.width / page_w, image.height / page_h)
    return boxes, (float(image.width), float(image.height)), scale


__all__ = [
    "BoundingBoxPixels",
    "OcrWord",
    "OcrBackend",
    "SUPPORTED_OCR_MODELS",
    "load_pdf_input",
    "load_image_input",
    "render_page_image",
    "page_image_size",
    "get_ocr_backend",
    "supported_backends",
    "correct_ocr_lines",
    "extract_native_words",
    "run_ocr_on_image",
    "extract_page_words",
    "resolve_device",
]
