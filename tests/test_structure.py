"""Tests for dynamic table reconstruction from word geometry."""

from pdf2xlsx.engines.opensource.layout import build_lines
from pdf2xlsx.engines.opensource.structure import (
    build_table_data,
    detect_page_tables,
)
from pdf2xlsx.pdf import Segment, WordBox
from pdf2xlsx.schemas import ExtractionResult, PageData
from pdf2xlsx.validation import validate_and_normalize

from conftest import BORDERLESS_TABLE, words_from_borderless_spec


def _validate(table):
    """Header propagation lives in the shared validation layer."""
    result = ExtractionResult(
        source_filename="t.pdf", page_count=1,
        pages=[PageData(page_number=1, tables=[table])],
    )
    result, report = validate_and_normalize(result)
    assert report.ok
    return result.pages[0].tables[0]


def w(text, x0, y0, width=40.0, height=10.0, conf=None):
    return WordBox(text=text, x0=x0, y0=y0, x1=x0 + width, y1=y0 + height, confidence=conf)


def _grid_segments(x0, y0, cols, rows, cw, ch):
    xs = [x0 + i * cw for i in range(cols + 1)]
    ys = [y0 + i * ch for i in range(rows + 1)]
    h = [Segment("h", y, xs[0], xs[-1]) for y in ys]
    v = [Segment("v", x, ys[0], ys[-1]) for x in xs]
    return h, v


def _make_rows(*row_texts):
    return [
        [w(t, col * 100.0 + 5.0, r * 20.0 + 5.0, width=60.0, height=9.0)
         for col, t in enumerate(cells) if t]
        for r, cells in enumerate(row_texts)
    ]


# ---------------------------------------------------------------------------
# build_table_data
# ---------------------------------------------------------------------------


def test_build_table_data_values_and_header():
    per_row_lines = [build_lines(words) for words in _make_rows(
        ["Head A", "Head B", "Head C"],
        ["1", "10", "100"],
        ["2", "20", "200"],
    )]
    x_bands = [(0.0, 100.0), (100.0, 200.0), (200.0, 300.0)]
    row_groups = [[line] for lines in per_row_lines for line in lines]
    table = build_table_data(
        page_number=1,
        table_index=0,
        x_bands=x_bands,
        row_groups=row_groups,
        space_w=600.0,
        space_h=800.0,
        source="whitespace",
    )
    assert table.n_columns == 3
    assert table.n_rows == 3
    assert table.rows[0].is_header is True
    assert 0.0 <= table.bbox.x0 < table.bbox.x1 <= 1.0
    assert 0.0 <= table.bbox.y0 < table.bbox.y1 <= 1.0

    validated = _validate(table)  # header names come from validation
    assert [c.header for c in validated.columns] == ["Head A", "Head B", "Head C"]
    assert [c.value for c in validated.rows[1].cells] == ["1", "10", "100"]


def test_build_table_data_all_text_rows_not_marked_header():
    per_row_lines = [build_lines(words) for words in _make_rows(
        ["Alpha", "Beta"],
        ["Gamma", "Delta"],
    )]
    x_bands = [(0.0, 100.0), (100.0, 200.0)]
    row_groups = [[line] for lines in per_row_lines for line in lines]
    table = build_table_data(
        page_number=1, table_index=0, x_bands=x_bands, row_groups=row_groups,
        space_w=600.0, space_h=800.0, source="whitespace",
    )
    # no numeric contrast -> conservative: first row stays data (no data loss)
    assert table.rows[0].is_header is False
    assert all(col.header is None for col in table.columns)


def test_build_table_data_empty_cells_and_confidence():
    lines = build_lines([
        w("h1", 5, 5, conf=0.9),
        w("h2", 105, 5, conf=0.7),
        w("v1", 5, 30, conf=0.8),
    ])
    x_bands = [(0.0, 100.0), (100.0, 200.0)]
    table = build_table_data(
        page_number=1, table_index=0, x_bands=x_bands, row_groups=[[ln] for ln in lines],
        space_w=200.0, space_h=200.0, source="whitespace",
    )
    assert table.rows[1].cells[1].value == ""  # empty cell present
    assert table.rows[0].cells[0].confidence == 0.9
    assert table.confidence is not None and 0.0 < table.confidence <= 1.0


def test_rtl_direction_propagates():
    rtl_lines = build_lines([w("سلام", 105, 5, width=50), w("جهان", 105, 30, width=50)])
    table = build_table_data(
        page_number=1, table_index=0,
        x_bands=[(0.0, 100.0), (100.0, 200.0)],
        row_groups=[[ln] for ln in rtl_lines],
        space_w=200.0, space_h=200.0, source="whitespace",
    )
    assert table.text_direction == "rtl"


# ---------------------------------------------------------------------------
# detect_page_tables (full page)
# ---------------------------------------------------------------------------


def test_detect_ruled_table_from_words_and_segments():
    spec_rows = [["ID", "Name", "Qty"], ["1", "Alpha", "5"], ["2", "Beta", "9"]]
    words = []
    for r, row in enumerate(spec_rows):
        y0 = 100.0 + r * 24.0
        for c, token in enumerate(row):
            words.append(w(token, 105.0 + c * 100.0, y0 + 6.0, width=45.0, height=9.0))
    h, v = _grid_segments(100.0, 100.0, cols=3, rows=3, cw=100.0, ch=24.0)
    tables = detect_page_tables(
        page_number=1, words=words, h_segments=h, v_segments=v,
        space_w=600.0, space_h=800.0,
    )
    assert len(tables) == 1
    t = _validate(tables[0])
    assert t.detection_source == "ruling_lines"
    assert t.n_rows == 3 and t.n_columns == 3
    assert [c.value for c in t.rows[0].cells] == ["ID", "Name", "Qty"]
    assert t.rows[0].is_header is True
    assert [c.header for c in t.columns] == ["ID", "Name", "Qty"]
    assert [c.value for c in t.rows[2].cells] == ["2", "Beta", "9"]


def test_detect_borderless_table_without_any_lines():
    words = words_from_borderless_spec(BORDERLESS_TABLE)
    tables = detect_page_tables(
        page_number=1, words=words, h_segments=[], v_segments=[],
        space_w=600.0, space_h=800.0,
    )
    assert len(tables) == 1
    t = tables[0]
    assert t.detection_source == "whitespace"
    assert t.n_rows == len(BORDERLESS_TABLE.rows)
    assert t.n_columns == 3
    assert [c.value for c in t.rows[1].cells] == ["A-1", "Alpha", "15"]


def test_no_tables_for_prose():
    words = []
    sentences = [
        ["The", "quick", "brown", "fox", "jumps"],
        ["A", "paragraph", "of", "running", "text"],
        ["Words", "are", "separated", "only", "by"],
        ["Additional", "sentences", "keep", "gaps", "uneven"],
    ]
    for r, sentence in enumerate(sentences):
        x = 50.0
        y = 60.0 + r * 18.0
        for token in sentence:
            words.append(w(token, x, y, width=5.0 * len(token), height=10.0))
            x += 5.0 * len(token) + 3.0  # ordinary word spacing, no aligned gaps
    tables = detect_page_tables(
        page_number=1, words=words, h_segments=[], v_segments=[],
        space_w=600.0, space_h=800.0,
    )
    assert tables == []


def test_grid_scale_applies_to_ocr_space():
    """Grid coordinates are in PDF points; OCR words are in pixels (2x)."""
    rows = [["A", "B"], ["1", "2"]]
    scale = 2.0
    words = []
    for r, row in enumerate(rows):
        for c, token in enumerate(row):
            words.append(
                w(token, (105.0 + c * 100.0) * scale,
                  (100.0 + r * 24.0 + 6.0) * scale,
                  width=45.0 * scale, height=9.0 * scale)
            )
    h, v = _grid_segments(100.0, 100.0, cols=2, rows=2, cw=100.0, ch=24.0)
    tables = detect_page_tables(
        page_number=1, words=words, h_segments=h, v_segments=v,
        space_w=1200.0, space_h=1600.0, grid_scale=(scale, scale),
    )
    assert len(tables) == 1
    assert [c.value for c in tables[0].rows[1].cells] == ["1", "2"]

