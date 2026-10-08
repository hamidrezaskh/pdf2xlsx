"""Tests for PDF loading and page processing."""

import pytest

from pdf2xlsx.exceptions import EmptyPdfError, InvalidPdfError, PdfPasswordError
from pdf2xlsx.pdf import load_pdf

from conftest import HAS_PERSIAN_FONT, build_empty_pdf, build_encrypted_pdf


def test_load_from_path(grid_pdf):
    pdf = load_pdf(grid_pdf)
    try:
        assert pdf.page_count == 1
        assert pdf.source_name == "grid.pdf"
        w, h = pdf.page_size(1)
        assert w > 100 and h > 100
    finally:
        pdf.close()


def test_load_from_bytes(grid_pdf):
    data = grid_pdf.read_bytes()
    pdf = load_pdf(data, source_name="from-bytes.pdf")
    try:
        assert pdf.page_count == 1
        assert pdf.source_name == "from-bytes.pdf"
        assert pdf.read_bytes()[:5] == b"%PDF-"
    finally:
        pdf.close()


def test_native_words(grid_pdf):
    with load_pdf(grid_pdf) as pdf:
        words = pdf.native_words(1)
        texts = {w.text for w in words}
        assert {"ID", "Name", "Qty", "Price"} <= texts
        assert all(w.x1 > w.x0 and w.y1 > w.y0 for w in words)


def test_render_png(grid_pdf):
    with load_pdf(grid_pdf) as pdf:
        png = pdf.render_png(1, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 500


def test_ruling_segments_detected(grid_pdf):
    with load_pdf(grid_pdf) as pdf:
        h_lines, v_lines = pdf.ruling_segments(1)
    assert len(h_lines) >= 4  # 4 rows -> >= 4 horizontal borders
    assert len(v_lines) >= 4  # 4 columns -> >= 4 vertical borders


def test_invalid_pdf_bytes():
    with pytest.raises(InvalidPdfError):
        load_pdf(b"this is not a pdf at all")


def test_empty_file():
    with pytest.raises(InvalidPdfError):
        load_pdf(b"")


def test_zero_page_pdf(tmp_path):
    try:
        path = build_empty_pdf(tmp_path / "empty.pdf")
    except ValueError:  # pragma: no cover - PyMuPDF refuses zero-page saves
        pytest.skip("cannot build a zero-page PDF with this PyMuPDF version")
    if path.stat().st_size == 0:  # pragma: no cover
        pytest.skip("zero-page PDF saved as empty file")
    with pytest.raises(EmptyPdfError):
        load_pdf(path)


def test_encrypted_pdf(tmp_path):
    path = build_encrypted_pdf(tmp_path / "locked.pdf")
    with pytest.raises(PdfPasswordError):
        load_pdf(path)


def test_missing_file(tmp_path):
    with pytest.raises(InvalidPdfError):
        load_pdf(tmp_path / "does-not-exist.pdf")


def test_page_out_of_range(grid_pdf):
    with load_pdf(grid_pdf) as pdf:
        with pytest.raises(IndexError):
            pdf.native_words(99)


@pytest.mark.skipif(not HAS_PERSIAN_FONT, reason="no Persian-capable font on this system")
def test_native_words_persian_logical_order(tmp_path):
    """RTL text must come back in logical order (shaping artifacts undone)."""
    from conftest import TableSpec, build_pdf

    spec = TableSpec(
        rows=[["ردیف", "شرح"], ["۱", "پرداخت"]],
        origin=(80.0, 80.0),
        cell_w=150.0,
        cell_h=26.0,
    )
    path = build_pdf(tmp_path / "fa_words.pdf", [[spec]])
    with load_pdf(path) as pdf:
        texts = {w.text for w in pdf.native_words(1)}
    assert "ردیف" in texts
    assert "شرح" in texts
    assert "پرداخت" in texts
    assert "۱" in texts
    # no presentation-form artifacts leaked through
    assert not any(0xFB50 <= ord(ch) <= 0xFEFF for t in texts for ch in t)


def test_native_words_groups_and_boxes(grid_pdf):
    with load_pdf(grid_pdf) as pdf:
        words = pdf.native_words(1)
    by_text = {w.text: w for w in words}
    box = by_text["Price"]
    assert box.x1 > box.x0 and box.y1 > box.y0
    assert box.cx > by_text["ID"].cx  # columns ordered left to right
    assert all(w.confidence is None for w in words)  # text layer is exact
