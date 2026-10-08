"""Tests for geometry-based layout detection (lines, grids, whitespace)."""

from pdf2xlsx.engines.opensource.layout import (
    TextLine,
    build_lines,
    detect_blocks,
    detect_grids,
    detect_whitespace_candidates,
    find_column_separators,
    first_strong_direction,
    page_direction,
)
from pdf2xlsx.pdf import Segment, WordBox

from conftest import (
    BORDERLESS_TABLE,
    DEFAULT_TABLE,
    words_from_borderless_spec,
    words_from_table_spec,
)


def w(text, x0, y0, width=40.0, height=10.0, conf=None):
    return WordBox(text=text, x0=x0, y0=y0, x1=x0 + width, y1=y0 + height, confidence=conf)


# ---------------------------------------------------------------------------
# lines & direction
# ---------------------------------------------------------------------------


def test_build_lines_groups_by_vertical_overlap():
    words = [
        w("one", 10, 10),
        w("two", 60, 11),   # same line
        w("three", 10, 30),  # next line
    ]
    lines = build_lines(words)
    assert len(lines) == 2
    assert {x.text for x in lines[0].words} == {"one", "two"}
    assert {x.text for x in lines[1].words} == {"three"}


def test_line_text_ltr_and_rtl_ordering():
    ltr = TextLine(words=[w("world", 60, 10), w("hello", 10, 10)], direction="ltr")
    assert ltr.text == "hello world"
    rtl = TextLine(words=[w("سلام", 60, 10), w("دنیا", 10, 10)], direction="rtl")
    assert rtl.text == "سلام دنیا"  # rightmost word comes first logically


def test_first_strong_direction():
    assert first_strong_direction("سلام") == "rtl"
    assert first_strong_direction("Hello") == "ltr"
    assert first_strong_direction("۱۲۳") == "ltr"  # digits are weak
    assert first_strong_direction("123 سلام") == "rtl"
    assert first_strong_direction("") == "ltr"


def test_page_direction_majority():
    lines = [
        TextLine(words=[w("سلام", 10, 10)], direction="rtl"),
        TextLine(words=[w("Hello", 10, 30)], direction="ltr"),
    ]
    assert page_direction(lines) == "ltr"  # tie -> ltr
    lines.append(TextLine(words=[w("جهان", 10, 50)], direction="rtl"))
    assert page_direction(lines) == "rtl"


# ---------------------------------------------------------------------------
# ruling-line grids
# ---------------------------------------------------------------------------


def _synthetic_grid_segments(x0=100.0, y0=100.0, cols=3, rows=4, cw=80.0, ch=20.0):
    xs = [x0 + i * cw for i in range(cols + 1)]
    ys = [y0 + i * ch for i in range(rows + 1)]
    h = [Segment("h", y, xs[0], xs[-1]) for y in ys]
    v = [Segment("v", x, ys[0], ys[-1]) for x in xs]
    return h, v


def test_detect_grids_from_synthetic_segments():
    h, v = _synthetic_grid_segments()
    grids = detect_grids(h, v)
    assert len(grids) == 1
    grid = grids[0]
    assert len(grid.x_edges) == 4
    assert len(grid.y_edges) == 5
    assert grid.bbox[0] == 100.0 and grid.bbox[1] == 100.0


def test_detect_grids_requires_minimum_structure():
    h = [Segment("h", 100.0, 50.0, 200.0), Segment("h", 120.0, 50.0, 200.0)]
    v = [Segment("v", 80.0, 90.0, 130.0)]
    assert detect_grids(h, v) == []  # only 1 column


def test_detect_grids_on_real_pdf(grid_pdf):
    from pdf2xlsx.pdf import load_pdf

    with load_pdf(grid_pdf) as pdf:
        h, v = pdf.ruling_segments(1)
    grids = detect_grids(h, v)
    assert len(grids) == 1
    assert len(grids[0].x_edges) == len(DEFAULT_TABLE.rows[0]) + 1
    assert len(grids[0].y_edges) == len(DEFAULT_TABLE.rows) + 1


# ---------------------------------------------------------------------------
# whitespace / borderless detection
# ---------------------------------------------------------------------------


def test_blocks_split_by_large_gaps():
    words = [w("a", 10, 10), w("b", 10, 24), w("c", 10, 200)]
    blocks = detect_blocks(build_lines(words))
    assert len(blocks) == 2
    assert len(blocks[0]) == 2 and len(blocks[1]) == 1


def test_find_separators_on_aligned_columns():
    words = []
    for row in range(4):
        y = 10 + row * 24
        words += [w(f"a{row}", 60, y, width=30), w(f"b{row}", 240, y, width=30), w(f"c{row}", 420, y, width=30)]
    lines = build_lines(words)
    separators = find_column_separators(lines, tolerance=6.0, min_gap=8.0)
    assert len(separators) == 2
    assert 150 < separators[0] < 220  # between col0 and col1
    assert 330 < separators[1] < 400  # between col1 and col2


def test_find_separators_rejects_prose():
    words = [
        w("words", 60, 10, width=40),
        w("have", 105, 10, width=30),
        w("varying", 140, 10, width=55),
        w("gaps", 200, 10, width=30),
        w("between", 60, 34, width=50),
        w("each", 115, 34, width=28),
        w("line", 148, 34, width=28),
        w("completely", 181, 34, width=70),
    ]
    lines = build_lines(words)
    separators = find_column_separators(lines, tolerance=6.0, min_gap=8.0)
    assert separators == []


def test_whitespace_candidates_from_borderless_spec():
    words = words_from_borderless_spec(BORDERLESS_TABLE)
    candidates = detect_whitespace_candidates(build_lines(words))
    assert len(candidates) == 1
    bands = candidates[0].x_bands
    assert len(bands) == len(BORDERLESS_TABLE.rows[0])  # 3 columns inferred
    assert len(candidates[0].lines) == len(BORDERLESS_TABLE.rows)
