"""Shared fixtures and PDF builders for the test suite.

Fixture PDFs are generated programmatically with PyMuPDF so tests are
fully offline and hermetic.  Nothing here hard-codes application logic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(ROOT))

import pymupdf as fitz
import pytest

from pdf2xlsx.config import Settings
from pdf2xlsx.pdf import WordBox

A4 = (595.0, 842.0)
PERSIAN_FONT = Path("C:/Windows/Fonts/tahoma.ttf")
HAS_PERSIAN_FONT = PERSIAN_FONT.exists()


# ---------------------------------------------------------------------------
# page specifications
# ---------------------------------------------------------------------------


@dataclass
class TableSpec:
    """A ruled table (grid lines drawn per cell)."""

    rows: List[List[str]]
    origin: Tuple[float, float] = (50.0, 60.0)
    cell_w: float = 120.0
    cell_h: float = 24.0


@dataclass
class BorderlessSpec:
    """A borderless table: aligned columns, no lines."""

    rows: List[List[str]]
    origins: List[float] = field(default_factory=lambda: [50.0, 230.0, 410.0])
    origin_y: float = 60.0
    line_pitch: float = 26.0


@dataclass
class ProseSpec:
    """Ordinary paragraph text (control: must NOT be detected as a table)."""

    lines: List[str]
    origin: Tuple[float, float] = (50.0, 60.0)
    line_pitch: float = 16.0


@dataclass
class TitleSpec:
    """A single line of text (e.g. a heading)."""

    text: str
    origin: Tuple[float, float] = (50.0, 40.0)
    fontsize: float = 14.0


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------


def _insert(page: fitz.Page, point, text: str, fontsize: float = 9.0, persian: bool = False) -> None:
    if persian and HAS_PERSIAN_FONT:
        # Route Arabic-script text through the HTML engine: it performs
        # proper bidi reordering + shaping, exactly like production PDF
        # writers (browsers, Word, ...), so fixtures stay realistic.
        x, y = point
        css = (
            f"@font-face {{font-family: fa; src: url({PERSIAN_FONT.as_posix()});}} "
            f"* {{font-family: fa; font-size: {fontsize}pt; margin: 0; padding: 0;}}"
        )
        rect = fitz.Rect(x, y - fontsize * 1.4, x + 300, y + fontsize * 0.9)
        page.insert_htmlbox(rect, text, css=css)
    else:
        page.insert_text(
            fitz.Point(*point), text, fontsize=fontsize, fontname="helv"
        )


def _draw_table(page: fitz.Page, spec: TableSpec) -> None:
    x0, y0 = spec.origin
    for r, row in enumerate(spec.rows):
        for c, value in enumerate(row):
            rect = fitz.Rect(
                x0 + c * spec.cell_w,
                y0 + r * spec.cell_h,
                x0 + (c + 1) * spec.cell_w,
                y0 + (r + 1) * spec.cell_h,
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            if value:
                persian = not value.isascii()
                _insert(
                    page,
                    (rect.x0 + 4, rect.y0 + spec.cell_h * 0.68),
                    value,
                    fontsize=9,
                    persian=persian,
                )


def _draw_borderless(page: fitz.Page, spec: BorderlessSpec) -> None:
    for r, row in enumerate(spec.rows):
        y = spec.origin_y + r * spec.line_pitch + 10
        for c, value in enumerate(row):
            if not value:
                continue
            persian = not value.isascii()
            _insert(page, (spec.origins[c], y), value, fontsize=10, persian=persian)


def _draw_prose(page: fitz.Page, spec: ProseSpec) -> None:
    x, y = spec.origin
    for i, line in enumerate(spec.lines):
        _insert(page, (x, y + i * spec.line_pitch), line, fontsize=10)


def build_pdf(path: Path, pages: Sequence[Sequence[object]]) -> Path:
    """Build a PDF; each page is a sequence of specs to draw on it."""
    doc = fitz.open()
    for page_specs in pages:
        page = doc.new_page(width=A4[0], height=A4[1])
        for spec in page_specs:
            if isinstance(spec, TableSpec):
                _draw_table(page, spec)
            elif isinstance(spec, BorderlessSpec):
                _draw_borderless(page, spec)
            elif isinstance(spec, ProseSpec):
                _draw_prose(page, spec)
            elif isinstance(spec, TitleSpec):
                _insert(page, spec.origin, spec.text, fontsize=spec.fontsize)
            else:  # pragma: no cover
                raise TypeError(f"unknown spec: {type(spec)!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def build_scanned_pdf(path: Path, pages: Sequence[Sequence[object]], dpi: int = 150) -> Path:
    """Render pages to images and re-embed them (no text layer left)."""
    source = build_pdf(path.with_suffix(".src.pdf"), pages)
    src = fitz.open(str(source))
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(str(path))
    out.close()
    src.close()
    source.unlink(missing_ok=True)
    return path


def build_empty_pdf(path: Path) -> Path:
    doc = fitz.open()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def build_encrypted_pdf(path: Path) -> Path:
    doc = fitz.open()
    doc.new_page()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(
        str(path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="secret",
        owner_pw="secret",
    )
    doc.close()
    return path


def words_from_table_spec(spec: TableSpec, scale: float = 1.0) -> List[WordBox]:
    """Approximate word boxes for a TableSpec (used for fake OCR results)."""
    words: List[WordBox] = []
    for r, row in enumerate(spec.rows):
        y0 = spec.origin[1] + r * spec.cell_h + 6
        for c, value in enumerate(row):
            if not value:
                continue
            x = spec.origin[0] + c * spec.cell_w + 4
            for token in str(value).split():
                width = max(6.0, 5.2 * len(token))
                words.append(
                    WordBox(
                        text=token,
                        x0=x * scale,
                        y0=y0 * scale,
                        x1=(x + width) * scale,
                        y1=(y0 + 9) * scale,
                    )
                )
                x += width + 3
    return words


def words_from_borderless_spec(spec: BorderlessSpec, scale: float = 1.0) -> List[WordBox]:
    words: List[WordBox] = []
    for r, row in enumerate(spec.rows):
        y0 = spec.origin_y + r * spec.line_pitch
        for c, value in enumerate(row):
            if not value:
                continue
            x = spec.origins[c]
            for token in str(value).split():
                width = max(6.0, 5.5 * len(token))
                words.append(
                    WordBox(
                        text=token,
                        x0=x * scale,
                        y0=y0 * scale,
                        x1=(x + width) * scale,
                        y1=(y0 + 11) * scale,
                    )
                )
                x += width + 3
    return words


class FakeOcrBackend:
    """Deterministic OCR stand-in returning precomputed word boxes."""

    name = "fake"

    def __init__(self, words_by_page: Optional[dict] = None) -> None:
        self.words_by_page = words_by_page or {}
        self.calls: List[Tuple[int, Tuple[int, int]]] = []

    def run(self, image, page_number: int) -> List[WordBox]:
        self.calls.append((page_number, (image.width, image.height)))
        return list(self.words_by_page.get(page_number, []))


DEFAULT_TABLE = TableSpec(
    rows=[
        ["ID", "Name", "Qty", "Price"],
        ["1", "Item A", "10", "4.50"],
        ["2", "Item B", "3", "12.00"],
        ["3", "", "7", "1.25"],
    ],
    origin=(60.0, 80.0),
    cell_w=110.0,
    cell_h=24.0,
)

OTHER_TABLE = TableSpec(
    rows=[
        ["Region", "Total"],
        ["North", "100"],
        ["South", "250"],
    ],
    origin=(70.0, 400.0),
    cell_w=150.0,
    cell_h=26.0,
)

BORDERLESS_TABLE = BorderlessSpec(
    rows=[
        ["Code", "Label", "Amount"],
        ["A-1", "Alpha", "15"],
        ["B-2", "Beta", "42"],
        ["C-3", "Gamma", "8"],
    ],
    origins=[60.0, 240.0, 420.0],
    origin_y=100.0,
    line_pitch=26.0,
)


@pytest.fixture
def make_settings():
    """Build Settings directly (never touching the real environment)."""

    def _make(**overrides) -> Settings:
        base = dict(ocr_model="native", log_level="WARNING", debug=False)
        base.update(overrides)
        return Settings(**base)

    return _make


@pytest.fixture
def grid_pdf(tmp_path) -> Path:
    """Single page, one ruled table."""
    return build_pdf(tmp_path / "grid.pdf", [[DEFAULT_TABLE]])


@pytest.fixture
def borderless_pdf(tmp_path) -> Path:
    return build_pdf(tmp_path / "borderless.pdf", [[BORDERLESS_TABLE]])


@pytest.fixture
def multi_page_pdf(tmp_path) -> Path:
    """Page 1: one ruled table. Page 2: a second ruled table + borderless table."""
    return build_pdf(
        tmp_path / "multi.pdf",
        [
            [TitleSpec("Quarterly Report"), DEFAULT_TABLE],
            [TitleSpec("Summary"), OTHER_TABLE, BORDERLESS_TABLE],
        ],
    )


@pytest.fixture
def prose_pdf(tmp_path) -> Path:
    prose = ProseSpec(
        lines=[
            "The quick brown fox jumps over the lazy dog again today",
            "A paragraph of running text should never look like a table",
            "Words are separated only by ordinary single spaces here",
            "Additional sentences keep the vertical gaps inconsistent",
        ],
        origin=(60.0, 80.0),
    )
    return build_pdf(tmp_path / "prose.pdf", [[prose]])



