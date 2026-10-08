"""Tests for UI preview helpers.

Regression: duplicate column headers crashed ``st.dataframe`` because
pyarrow rejects DataFrames with non-unique labels.
"""

import py_compile
from pathlib import Path

import pytest

from pdf2xlsx.preview import build_preview_frame, unique_labels
from pdf2xlsx.schemas import Cell, ColumnData, RowData, TableData

ROOT = Path(__file__).resolve().parents[1]


def test_app_module_compiles():
    py_compile.compile(str(ROOT / "app.py"), doraise=True)


def test_unique_labels_keeps_singletons():
    assert unique_labels(["ID", "Name"]) == ["ID", "Name"]
    assert unique_labels(["ردیف", "شرح"]) == ["ردیف", "شرح"]


def test_unique_labels_deduplicates_repeats_and_empties():
    assert unique_labels(["جمع", "جمع", "", "", "جمع"]) == [
        "جمع",
        "جمع (2)",
        "",
        "(2)",
        "جمع (3)",
    ]


def _table(headers):
    n = len(headers)
    return TableData(
        columns=[ColumnData(index=i, header=h or None) for i, h in enumerate(headers)],
        rows=[
            RowData(
                index=0,
                cells=[
                    Cell(row_index=0, column_index=i, value=f"v{i}") for i in range(n)
                ],
            )
        ],
    )


def test_build_preview_frame_duplicate_headers():
    frame = build_preview_frame(_table(["جمع", "جمع", "", ""]))
    assert frame.columns.is_unique  # pyarrow requirement
    assert list(frame.iloc[0]) == ["v0", "v1", "v2", "v3"]


def test_build_preview_frame_headerless_uses_default_columns():
    frame = build_preview_frame(_table(["", ""]))  # all blank -> no header row
    assert frame.columns.is_unique
    assert frame.shape == (1, 2)


def test_build_preview_frame_unicode_values():
    table = TableData(
        columns=[ColumnData(index=0, header="شرح")],
        rows=[
            RowData(
                index=0,
                cells=[Cell(row_index=0, column_index=0, value="پرداخت آب")],
            )
        ],
    )
    frame = build_preview_frame(table)
    assert frame.iloc[0, 0] == "پرداخت آب"


def test_frame_survives_pyarrow_conversion():
    """Exact crash from the traceback: pyarrow must accept the frame now."""
    pa = pytest.importorskip("pyarrow")
    frame = build_preview_frame(_table(["A", "A", ""]))
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    assert arrow.num_rows == 1
    assert arrow.num_columns == 3
