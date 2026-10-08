"""PDF loading and page processing (PyMuPDF).

Provides:
* robust PDF loading with friendly errors (invalid / encrypted / empty)
* native text extraction (word boxes with coordinates)
* vector ruling-line extraction (for table grid detection)
* page rendering to images (for OCR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

try:
    import pymupdf as fitz  # modern PyMuPDF (>= 1.24)
except ImportError:  # pragma: no cover - legacy fallback
    import fitz  # type: ignore

from ..exceptions import EmptyPdfError, InvalidPdfError, PdfPasswordError
from ..textutil import assemble_word_chars, line_base_direction

logger = logging.getLogger(__name__)

# Clustering tolerance for ruling lines, in PDF points.
LINE_CLUSTER_TOL = 2.0

PdfSource = Union[str, Path, bytes, bytearray]


@dataclass(frozen=True)
class WordBox:
    """A word with its bounding box in page coordinates."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float | None = None

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)


@dataclass(frozen=True)
class Segment:
    """An axis-aligned ruling segment (PDF points).

    ``position`` is the y coordinate for horizontal segments and the x
    coordinate for vertical ones; ``start``/``end`` span the other axis.
    """

    orientation: str  # "h" or "v"
    position: float
    start: float
    end: float


def load_pdf(source: PdfSource, *, source_name: str | None = None) -> "PdfDocument":
    """Load and validate a PDF from bytes or a filesystem path."""
    data: bytes | None = None
    path: Path | None = None

    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        name = source_name or "document.pdf"
        if not data:
            raise InvalidPdfError("The uploaded file is empty.")
        if not data.lstrip()[:5].startswith(b"%PDF"):
            raise InvalidPdfError(
                "The uploaded file is not a valid PDF document.",
                detail=f"missing PDF header, first bytes: {data[:8]!r}",
            )
    else:
        path = Path(source)
        name = source_name or path.name
        if not path.exists():
            raise InvalidPdfError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise InvalidPdfError("The uploaded file is empty.")
        data = path.read_bytes()

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError(
            "The file could not be opened as a PDF. It may be corrupted.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        if doc.needs_pass:
            doc.close()
            raise PdfPasswordError(
                "The PDF is password protected. Please remove the password and try again."
            )
        if doc.page_count == 0:
            doc.close()
            raise EmptyPdfError("The PDF contains no pages.")
    except (PdfPasswordError, EmptyPdfError):
        raise

    logger.info("PDF loaded: %s (%d page(s))", name, doc.page_count)
    return PdfDocument(doc, source_name=name, data=data)


class PdfDocument:
    """Thin wrapper around a fitz.Document with 1-based page access."""

    def __init__(self, doc: fitz.Document, *, source_name: str, data: bytes | None = None) -> None:
        self._doc = doc
        self.source_name = source_name
        self._data = data
        self._closed = False

    # -- basic properties -------------------------------------------------
    @property
    def page_count(self) -> int:
        return self._doc.page_count

    @property
    def is_closed(self) -> bool:
        return self._closed

    def read_bytes(self) -> bytes:
        if self._data is None:
            self._data = self._doc.tobytes(garbage=3, deflate=True)
        return self._data

    # -- context management ----------------------------------------------
    def close(self) -> None:
        if not self._closed:
            try:
                self._doc.close()
            finally:
                self._closed = True

    def __enter__(self) -> "PdfDocument":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- page helpers -----------------------------------------------------
    def _page(self, page_number: int) -> fitz.Page:
        if page_number < 1 or page_number > self.page_count:
            raise IndexError(f"page {page_number} out of range (1..{self.page_count})")
        return self._doc.load_page(page_number - 1)

    def page_size(self, page_number: int) -> Tuple[float, float]:
        rect = self._page(page_number).rect
        return (float(rect.width), float(rect.height))


    # -- text extraction --------------------------------------------------
    def native_words(self, page_number: int) -> List[WordBox]:
        """Words from the embedded text layer (exact, zero-cost).

        Words are rebuilt from per-character bounding boxes: glyph
        positions in a PDF are always visual, so sorting a word's
        characters by x gives the visual order, which is then converted
        to logical reading order (see :func:`pdf2xlsx.textutil`).
        This correctly handles shaped/presentation-form Arabic-script
        text as well as LTR content, without any destructive normalization.
        """
        page = self._page(page_number)
        data = page.get_text("rawdict")
        words: List[WordBox] = []

        def flush(
            group: list[tuple[float, float, float, float, str]], line_dir: str
        ) -> None:
            if not group:
                return
            text = assemble_word_chars([item[4] for item in group], base_dir=line_dir)
            if not text.strip():
                return
            words.append(
                WordBox(
                    text=text,
                    x0=min(item[0] for item in group),
                    y0=min(item[1] for item in group),
                    x1=max(item[2] for item in group),
                    y1=max(item[3] for item in group),
                )
            )

        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:  # image blocks carry no text
                continue
            for line in block.get("lines", []):
                chars: list[tuple[float, float, float, float, str]] = []
                for span in line.get("spans", []):
                    for ch in span.get("chars", []):
                        glyph = ch.get("c") or ch.get("unicode") or ""
                        if not glyph:
                            continue
                        bx0, by0, bx1, by1 = ch.get("bbox", (0, 0, 0, 0))
                        chars.append(
                            (float(bx0), float(by0), float(bx1), float(by1), glyph)
                        )
                if not chars:
                    continue
                # majority direction of the whole line (reliable bidi context)
                line_dir = line_base_direction([item[4] for item in chars])
                chars.sort(key=lambda item: item[0])  # visual (x) order

                group: list[tuple[float, float, float, float, str]] = []
                prev: tuple[float, float, float, float, str] | None = None
                for item in chars:
                    if item[4].isspace():
                        flush(group, line_dir)
                        group = []
                        prev = None
                        continue
                    if prev is not None:
                        gap = item[0] - prev[2]
                        height = max(1.0, item[3] - item[1])
                        if gap > 0.6 * height:  # wide gap without space -> new word
                            flush(group, line_dir)
                            group = []
                    group.append(item)
                    prev = item
                flush(group, line_dir)
        return words

    def has_text_layer(self, page_number: int, min_words: int = 5) -> bool:
        return len(self.native_words(page_number)) >= min_words

    # -- ruling lines -----------------------------------------------------
    def ruling_segments(self, page_number: int) -> Tuple[List[Segment], List[Segment]]:
        """Clustered horizontal/vertical ruling segments (PDF points).

        These come from vector drawings (table borders / grid lines) and
        are used for rule-based table grid detection.
        """
        page = self._page(page_number)
        raw_h: list[tuple[float, float, float]] = []
        raw_v: list[tuple[float, float, float]] = []
        tol = LINE_CLUSTER_TOL

        for drawing in page.get_drawings():
            for item in drawing["items"]:
                tag = item[0]
                if tag == "l":
                    p1, p2 = item[1], item[2]
                    if abs(p1.y - p2.y) <= 0.75:
                        raw_h.append((float(p1.y), min(p1.x, p2.x), max(p1.x, p2.x)))
                    elif abs(p1.x - p2.x) <= 0.75:
                        raw_v.append((float(p1.x), min(p1.y, p2.y), max(p1.y, p2.y)))
                elif tag == "re":
                    rect = fitz.Rect(item[1])
                    w, h = float(rect.width), float(rect.height)
                    if w <= 2.0 and h <= 2.0:
                        continue
                    if h <= 2.0:  # thin horizontal bar acts as a line
                        raw_h.append((float(rect.y0) + h / 2.0, rect.x0, rect.x1))
                    elif w <= 2.0:  # thin vertical bar acts as a line
                        raw_v.append((float(rect.x0) + w / 2.0, rect.y0, rect.y1))
                    else:  # full rectangle: its four edges are candidate lines
                        raw_h.append((float(rect.y0), rect.x0, rect.x1))
                        raw_h.append((float(rect.y1), rect.x0, rect.x1))
                        raw_v.append((float(rect.x0), rect.y0, rect.y1))
                        raw_v.append((float(rect.x1), rect.y0, rect.y1))

        return _cluster_segments(raw_h, tol, "h"), _cluster_segments(raw_v, tol, "v")

    # -- rendering --------------------------------------------------------
    def render_png(self, page_number: int, dpi: int = 200) -> bytes:
        """Render a page to PNG bytes (used for OCR)."""
        page = self._page(page_number)
        pix = page.get_pixmap(dpi=int(dpi))
        return pix.tobytes("png")


def _cluster_segments(
    raw: Sequence[tuple[float, float, float]], tol: float, orientation: str
) -> List[Segment]:
    """Group collinear raw segments and merge overlapping spans."""
    if not raw:
        return []
    ordered = sorted(raw, key=lambda r: r[0])
    groups: list[list[tuple[float, float, float]]] = []
    group_pos: list[float] = []
    current: list[tuple[float, float, float]] = []
    current_positions: list[float] = []

    for pos, start, end in ordered:
        if current and pos - (sum(current_positions) / len(current_positions)) > tol:
            groups.append(current)
            group_pos.append(sum(current_positions) / len(current_positions))
            current, current_positions = [], []
        current.append((pos, start, end))
        current_positions.append(pos)
    if current:
        groups.append(current)
        group_pos.append(sum(current_positions) / len(current_positions))

    segments: List[Segment] = []
    for items, position in zip(groups, group_pos):
        spans = sorted((min(s, e), max(s, e)) for _, s, e in items)
        merged: list[list[float]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 2 * tol:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            segments.append(Segment(orientation=orientation, position=position, start=start, end=end))
    return segments


__all__ = ["WordBox", "Segment", "PdfDocument", "load_pdf", "PdfSource"]

