"""Tests for the normalized structured data model."""

from pdf2xlsx.schemas import (
    Cell,
    ColumnData,
    ExtractionPayload,
    ExtractionResult,
    MergedCell,
    PageData,
    RowData,
    TableData,
)


def make_result() -> ExtractionResult:
    table = TableData(
        table_index=0,
        page_number=1,
        columns=[ColumnData(index=0, header="Name"), ColumnData(index=1, header="Qty")],
        rows=[
            RowData(
                index=0,
                is_header=True,
                cells=[
                    Cell(row_index=0, column_index=0, value="Name"),
                    Cell(row_index=0, column_index=1, value="Qty"),
                ],
            ),
            RowData(
                index=1,
                cells=[
                    Cell(row_index=1, column_index=0, value="قلم"),
                    Cell(row_index=1, column_index=1, value="42"),
                ],
            ),
        ],
        text_direction="rtl",
    )
    return ExtractionResult(
        source_filename="doc.pdf",
        page_count=1,
        processing_mode="openai",
        pages=[PageData(page_number=1, tables=[table])],
    )


def test_defaults():
    cell = Cell(row_index=0, column_index=0)
    assert cell.value == ""
    assert cell.rowspan == 1 and cell.colspan == 1
    assert cell.bbox is None and cell.confidence is None
    table = TableData()
    assert table.columns == [] and table.rows == []
    assert table.text_direction == "ltr"


def test_round_trip_json():
    result = make_result()
    restored = ExtractionResult.model_validate_json(result.model_dump_json())
    assert restored.pages[0].tables[0].rows[1].cells[0].value == "قلم"
    assert restored.pages[0].tables[0].text_direction == "rtl"


def test_iter_tables_and_counts():
    result = make_result()
    pairs = list(result.iter_tables())
    assert len(pairs) == 1
    page, table = pairs[0]
    assert page.page_number == 1
    assert table.n_rows == 2
    assert table.n_columns == 2
    assert result.table_count == 1


def test_to_matrix_excludes_header_row_when_headers_exist():
    headers, body = make_result().pages[0].tables[0].to_matrix()
    assert headers == ["Name", "Qty"]
    assert body == [["قلم", "42"]]


def test_to_matrix_includes_header_row_without_column_headers():
    table = TableData(
        columns=[ColumnData(index=0), ColumnData(index=1)],
        rows=[
            RowData(index=0, is_header=True, cells=[
                Cell(row_index=0, column_index=0, value="H1"),
                Cell(row_index=0, column_index=1, value="H2"),
            ]),
            RowData(index=1, cells=[
                Cell(row_index=1, column_index=0, value="a"),
                Cell(row_index=1, column_index=1, value="b"),
            ]),
        ],
    )
    headers, body = table.to_matrix()
    assert headers == ["", ""]
    assert body == [["H1", "H2"], ["a", "b"]]


def test_extraction_payload_accepts_pages_only():
    payload = ExtractionPayload.model_validate(
        {"pages": [{"page_number": 2, "tables": [{"table_index": 0}]}]}
    )
    assert payload.pages[0].page_number == 2
    assert payload.pages[0].tables[0].rows == []


def test_merged_cell_schema():
    merged = MergedCell(row_index=0, column_index=1, rowspan=2, colspan=2, value="x")
    assert merged.rowspan == 2
    dumped = merged.model_dump()
    assert dumped["column_index"] == 1
