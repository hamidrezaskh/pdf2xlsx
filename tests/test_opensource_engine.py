"""Tests for the open-source OCR / document-understanding engine."""

import importlib.util

import pytest

from pdf2xlsx.config import Settings
from pdf2xlsx.engines.opensource import OpenSourceEngine
from pdf2xlsx.engines.opensource.ocr import (
    create_ocr_backend,
    resolve_device,
)
from pdf2xlsx.exceptions import ConfigurationError, OcrError
from pdf2xlsx.pdf import load_pdf

from conftest import (
    BORDERLESS_TABLE,
    DEFAULT_TABLE,
    FakeOcrBackend,
    build_pdf,
    build_scanned_pdf,
    words_from_borderless_spec,
    words_from_table_spec,
)

EASYOCR_INSTALLED = importlib.util.find_spec("easyocr") is not None


def _extract(pdf_path, settings, ocr_backend=None):
    engine = OpenSourceEngine(settings, ocr_backend=ocr_backend)
    with load_pdf(pdf_path) as pdf:
        return engine.extract(pdf)


# ---------------------------------------------------------------------------
# native text-layer path
# ---------------------------------------------------------------------------


def test_ruled_table_from_text_layer(grid_pdf, make_settings):
    result = _extract(grid_pdf, make_settings())
    assert result.processing_mode == "opensource"
    assert result.page_count == 1
    assert result.table_count == 1
    page, table = next(result.iter_tables())
    assert page.page_number == 1
    assert table.detection_source == "ruling_lines"
    assert [c.value for c in table.rows[0].cells] == ["ID", "Name", "Qty", "Price"]
    assert [c.value for c in table.rows[3].cells] == ["3", "", "7", "1.25"]  # empty cell


def test_borderless_table_via_whitespace(borderless_pdf, make_settings):
    result = _extract(borderless_pdf, make_settings())
    assert result.table_count == 1
    _, table = next(result.iter_tables())
    assert table.detection_source == "whitespace"
    assert table.n_columns == 3
    assert table.n_rows == 4


def test_multi_page_multiple_tables(multi_page_pdf, make_settings):
    result = _extract(multi_page_pdf, make_settings())
    assert result.page_count == 2
    assert result.table_count == 3
    pages = [p.page_number for p in result.pages]
    assert pages == [1, 2]
    assert [len(p.tables) for p in result.pages] == [1, 2]


def test_prose_not_detected_as_table(prose_pdf, make_settings):
    result = _extract(prose_pdf, make_settings())
    assert result.table_count == 0
    assert result.warnings == []


def test_dynamic_column_counts_per_table(make_settings, tmp_path):
    wide = type(DEFAULT_TABLE)(
        rows=[["A", "B", "C", "D", "E"], ["1", "2", "3", "4", "5"]],
        origin=(50.0, 60.0), cell_w=90.0, cell_h=22.0,
    )
    narrow = type(DEFAULT_TABLE)(
        rows=[["X", "Y"], ["9", "8"]],
        origin=(50.0, 400.0), cell_w=140.0, cell_h=22.0,
    )
    path = build_pdf(tmp_path / "dynamic.pdf", [[wide, narrow]])
    result = _extract(path, make_settings())
    assert result.table_count == 2
    counts = [t.n_columns for _, t in result.iter_tables()]
    assert counts == [5, 2]


# ---------------------------------------------------------------------------
# OCR path (scanned PDFs)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(EASYOCR_INSTALLED, reason="easyocr installed; using real OCR instead")
def test_scanned_pdf_uses_injected_ocr_backend(make_settings, tmp_path):
    scanned = build_scanned_pdf(tmp_path / "scanned.pdf", [[BORDERLESS_TABLE]])
    scale = 150.0 / 72.0  # build_scanned_pdf renders at 150 dpi
    words = {1: words_from_borderless_spec(BORDERLESS_TABLE, scale=scale)}
    fake = FakeOcrBackend(words)
    result = _extract(scanned, make_settings(ocr_model="fake"), ocr_backend=fake)
    assert fake.calls and fake.calls[0][0] == 1
    assert result.table_count == 1
    _, table = next(result.iter_tables())
    assert table.n_columns == 3
    assert [c.value for c in table.rows[1].cells] == ["A-1", "Alpha", "15"]


@pytest.mark.skipif(EASYOCR_INSTALLED, reason="easyocr installed; using real OCR instead")
def test_scanned_pdf_with_ocr_disabled_warns(make_settings, tmp_path):
    scanned = build_scanned_pdf(tmp_path / "scanned.pdf", [[BORDERLESS_TABLE]])
    result = _extract(scanned, make_settings(ocr_model="native"))
    assert result.table_count == 0
    assert any("no text layer" in w for w in result.warnings)


@pytest.mark.skipif(EASYOCR_INSTALLED, reason="easyocr installed; using real OCR instead")
def test_scanned_pdf_ocr_not_installed_warns(make_settings, tmp_path):
    scanned = build_scanned_pdf(tmp_path / "scanned.pdf", [[BORDERLESS_TABLE]])
    result = _extract(scanned, make_settings(ocr_model="easyocr"))
    assert result.table_count == 0
    assert any("not installed" in w for w in result.warnings)


def test_ocr_model_cached_across_requests(make_settings):
    """The OCR backend is resolved once and reused between extractions."""
    engine = OpenSourceEngine(make_settings())
    first = engine._get_ocr()
    second = engine._get_ocr()
    assert first is second


# ---------------------------------------------------------------------------
# OCR backend factory
# ---------------------------------------------------------------------------


def test_unknown_ocr_model_raises(make_settings):
    with pytest.raises(ConfigurationError):
        create_ocr_backend(make_settings(ocr_model="does-not-exist"))


def test_native_ocr_model_disables_ocr(make_settings):
    assert create_ocr_backend(make_settings(ocr_model="native")) is None


def test_easyocr_backend_created(make_settings):
    if EASYOCR_INSTALLED:  # pragma: no cover
        pytest.skip("easyocr installed")
    backend = create_ocr_backend(make_settings(ocr_model="easyocr"))
    assert backend.name == "easyocr"
    assert backend.languages == ("fa", "en")
    with pytest.raises(OcrError):
        backend.run(None, 1)  # missing dependency -> friendly error


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
