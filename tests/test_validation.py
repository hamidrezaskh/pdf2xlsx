"""Tests for validation / normalization of structured output."""

from pdf2xlsx.schemas import (
    Cell,
    ColumnData,
    ExtractionResult,
    PageData,
    RowData,
    TableData,
)
from pdf2xlsx.validation import is_numeric_text, validate_and_normalize


def _result(*tables) -> ExtractionResult:
    """One page containing the given tables."""
    pages = [PageData(page_number=1, tables=list(tables))]
    return ExtractionResult(source_filename="x.pdf", page_count=1, pages=pages)


def test_ragged_rows_are_padded():
    table = TableData(
        rows=[
            RowData(index=0, cells=[Cell(row_index=0, column_index=0, value="a"),
                                    Cell(row_index=0, column_index=1, value="b"),
                                    Cell(row_index=0, column_index=2, value="c")]),
            RowData(index=1, cells=[Cell(row_index=1, column_index=0, value="d")]),
        ]
    )
    result, report = validate_and_normalize(_result(table))
    row1 = result.pages[0].tables[0].rows[1]
    assert [c.value for c in row1.cells] == ["d", "", ""]
    assert len(result.pages[0].tables[0].columns) == 3
    assert report.ok


def test_columns_inferred_from_cells():
    table = TableData(
        rows=[RowData(index=0, cells=[
            Cell(row_index=0, column_index=0, value="a"),
            Cell(row_index=0, column_index=3, value="d"),
        ])]
    )
    result, report = validate_and_normalize(_result(table))
    t = result.pages[0].tables[0]
    assert len(t.columns) == 4
    assert len(t.rows[0].cells) == 4
    assert report.ok


def test_row_indexes_made_contiguous():
    table = TableData(
        rows=[
            RowData(index=5, cells=[Cell(row_index=5, column_index=0, value="x")]),
            RowData(index=9, cells=[Cell(row_index=9, column_index=0, value="y")]),
        ]
    )
    result, report = validate_and_normalize(_result(table))
    rows = result.pages[0].tables[0].rows
    assert [r.index for r in rows] == [0, 1]
    assert [r.cells[0].value for r in rows] == ["x", "y"]
    assert any(i.code == "row_index_remapped" for i in report.warnings)


def test_duplicate_and_invalid_cells():
    table = TableData(
        rows=[RowData(index=0, cells=[
            Cell(row_index=0, column_index=0, value="first"),
            Cell(row_index=0, column_index=0, value="second"),
            Cell(row_index=0, column_index=-1, value="bad"),
        ])]
    )
    result, report = validate_and_normalize(_result(table))
    cells = result.pages[0].tables[0].rows[0].cells
    assert cells[0].value == "first"
    assert any(i.code == "duplicate_cell" for i in report.warnings)
    assert any(i.code == "invalid_cell_index" for i in report.errors)


def test_header_row_propagates_to_columns():
    table = TableData(
        rows=[
            RowData(index=0, is_header=True, cells=[
                Cell(row_index=0, column_index=0, value="Name"),
                Cell(row_index=0, column_index=1, value="Qty"),
            ]),
            RowData(index=1, cells=[
                Cell(row_index=1, column_index=0, value="x"),
                Cell(row_index=1, column_index=1, value="1"),
            ]),
        ]
    )
    result, _ = validate_and_normalize(_result(table))
    t = result.pages[0].tables[0]
    assert [c.header for c in t.columns] == ["Name", "Qty"]


def test_merged_cells_derived_from_spans_and_covered_cleared():
    table = TableData(
        rows=[
            RowData(index=0, cells=[
                Cell(row_index=0, column_index=0, value="top", colspan=2),
                Cell(row_index=0, column_index=1, value="should-clear"),
            ]),
            RowData(index=1, cells=[
                Cell(row_index=1, column_index=0, value="a"),
                Cell(row_index=1, column_index=1, value="b"),
            ]),
        ]
    )
    result, report = validate_and_normalize(_result(table))
    t = result.pages[0].tables[0]
    assert len(t.merged_cells) == 1
    merged = t.merged_cells[0]
    assert (merged.row_index, merged.column_index, merged.rowspan, merged.colspan) == (0, 0, 1, 2)
    assert t.rows[0].cells[0].value == "top"
    assert t.rows[0].cells[1].value == ""  # covered value cleared
    assert any(i.code == "covered_value_cleared" for i in report.warnings)


def test_spans_clamped_to_table_bounds():
    table = TableData(
        rows=[RowData(index=0, cells=[
            Cell(row_index=0, column_index=0, value="v", rowspan=10, colspan=10),
        ])]
    )
    result, _ = validate_and_normalize(_result(table))
    cell = result.pages[0].tables[0].rows[0].cells[0]
    assert cell.rowspan == 1 and cell.colspan == 1  # table is 1x1


def test_confidence_clamped():
    table = TableData(
        confidence=3.5,
        rows=[RowData(index=0, cells=[
            Cell(row_index=0, column_index=0, value="v", confidence=-1.0),
        ])],
    )
    result, report = validate_and_normalize(_result(table))
    t = result.pages[0].tables[0]
    assert t.confidence == 1.0
    assert t.rows[0].cells[0].confidence == 0.0
    assert any(i.code == "confidence_out_of_range" for i in report.warnings)


def test_pages_sorted_and_duplicates_merged():
    table_a = TableData(table_index=0, rows=[RowData(index=0, cells=[
        Cell(row_index=0, column_index=0, value="p2")])])
    table_a.page_number = 2
    table_b = TableData(table_index=0, rows=[RowData(index=0, cells=[
        Cell(row_index=0, column_index=0, value="p1")])])
    table_b.page_number = 1
    result = ExtractionResult(
        source_filename="x.pdf",
        page_count=2,
        pages=[
            PageData(page_number=2, tables=[table_a]),
            PageData(page_number=1, tables=[table_b]),
        ],
    )
    result, report = validate_and_normalize(result)
    assert [p.page_number for p in result.pages] == [1, 2]
    assert report.ok


def test_empty_table_warns_but_keeps():
    result, report = validate_and_normalize(_result(TableData()))
    assert any(i.code == "empty_table" for i in report.warnings)
    assert report.ok


def test_is_numeric_text_variants():
    assert is_numeric_text("42")
    assert is_numeric_text("4.50")
    assert is_numeric_text("۱۲۳")  # Persian digits
    assert is_numeric_text("1,250.00")
    assert not is_numeric_text("Item A")
    assert not is_numeric_text("")
    assert not is_numeric_text("42 kg")

