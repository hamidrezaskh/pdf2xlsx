"""Validation and normalization of engine output.

The same rules apply to every engine: row/column indexes are made
contiguous, ragged rows are padded, duplicate or invalid cells are
reported, spans are clamped, merged-cell regions are derived from spans
and header rows are propagated to column names.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from .schemas import (
    Cell,
    ColumnData,
    ExtractionResult,
    MergedCell,
    PageData,
    RowData,
    TableData,
)

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    location: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience
        loc = f" @ {self.location}" if self.location else ""
        return f"[{self.severity}] {self.code}{loc}: {self.message}"


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, location: str = "") -> None:
        self.issues.append(ValidationIssue(severity, code, message, location))

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def summary(self) -> str:
        return f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"


def _clamp_confidence(
    value: Optional[float], report: ValidationReport, loc: str
) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        report.add(
            SEVERITY_WARNING, "invalid_confidence", f"non-numeric confidence {value!r}", loc
        )
        return None
    if v < 0.0 or v > 1.0:
        report.add(
            SEVERITY_WARNING,
            "confidence_out_of_range",
            f"confidence {v} clamped to 0..1",
            loc,
        )
        return max(0.0, min(1.0, v))
    return v


def _normalize_table(table: TableData, report: ValidationReport, loc: str) -> None:
    # --- rows: sort and make indexes contiguous -------------------------
    rows = sorted(table.rows, key=lambda r: r.index)
    for new_index, row in enumerate(rows):
        if row.index != new_index:
            report.add(
                SEVERITY_WARNING,
                "row_index_remapped",
                f"row index {row.index} remapped to {new_index}",
                loc,
            )
            row.index = new_index
    table.rows = rows

    # --- column count (dynamic: from declared columns and actual cells) --
    max_cell_col = max((c.column_index for row in rows for c in row.cells), default=-1)
    n_cols = max(len(table.columns), max_cell_col + 1)
    while len(table.columns) < n_cols:
        table.columns.append(ColumnData(index=len(table.columns)))
    for index, col in enumerate(table.columns):
        if col.index != index:
            report.add(
                SEVERITY_WARNING,
                "column_index_remapped",
                f"column index {col.index} remapped to {index}",
                loc,
            )
            col.index = index

    # --- cells: fix indexes, deduplicate, pad ragged rows ----------------
    for row in rows:
        seen: dict[int, Cell] = {}
        for cell in sorted(row.cells, key=lambda c: c.column_index):
            if cell.row_index != row.index:
                report.add(
                    SEVERITY_WARNING,
                    "cell_row_mismatch",
                    f"cell row_index {cell.row_index} corrected to {row.index}",
                    loc,
                )
                cell.row_index = row.index
            if cell.column_index < 0:
                report.add(
                    SEVERITY_ERROR,
                    "invalid_cell_index",
                    f"negative column index {cell.column_index} dropped",
                    loc,
                )
                continue
            if cell.column_index in seen:
                report.add(
                    SEVERITY_WARNING,
                    "duplicate_cell",
                    f"duplicate cell at row {row.index}, column {cell.column_index}; keeping first",
                    loc,
                )
                continue
            cell.confidence = _clamp_confidence(cell.confidence, report, loc)
            seen[cell.column_index] = cell
        for col_index in range(n_cols):
            if col_index not in seen:
                seen[col_index] = Cell(row_index=row.index, column_index=col_index, value="")
        row.cells = [seen[i] for i in range(n_cols)]

    # --- spans: clamp to table bounds ------------------------------------
    n_rows = len(rows)
    for row in rows:
        for cell in row.cells:
            cell.rowspan = max(1, min(int(cell.rowspan or 1), max(1, n_rows - row.index)))
            cell.colspan = max(
                1, min(int(cell.colspan or 1), max(1, n_cols - cell.column_index))
            )

    # --- header rows propagate into column names -------------------------
    header_rows = [row for row in rows if row.is_header]
    for row in header_rows:
        for cell in row.cells:
            col = table.columns[cell.column_index]
            if not col.header and cell.value.strip():
                col.header = cell.value
    if len(header_rows) > 1:
        report.add(
            SEVERITY_WARNING,
            "multiple_header_rows",
            f"{len(header_rows)} header rows marked in one table",
            loc,
        )

    _derive_merged_cells(table, rows, report, loc)
    table.confidence = _clamp_confidence(table.confidence, report, loc)

    if not rows and not table.columns:
        report.add(SEVERITY_WARNING, "empty_table", "table has no rows and no columns", loc)


def _derive_merged_cells(
    table: TableData, rows: List[RowData], report: ValidationReport, loc: str
) -> None:
    """Build merged-cell regions from cell spans and clear covered cells."""
    merged: List[MergedCell] = []
    covered: set[tuple[int, int]] = set()
    for row in rows:
        for cell in row.cells:
            if cell.rowspan <= 1 and cell.colspan <= 1:
                continue
            anchor = (row.index, cell.column_index)
            if anchor in covered:
                report.add(
                    SEVERITY_WARNING,
                    "overlapping_merge",
                    f"overlapping merged cell at {anchor} skipped",
                    loc,
                )
                continue
            merged.append(
                MergedCell(
                    row_index=row.index,
                    column_index=cell.column_index,
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    value=cell.value,
                    bbox=cell.bbox,
                )
            )
            for r in range(row.index, row.index + cell.rowspan):
                for c in range(cell.column_index, cell.column_index + cell.colspan):
                    covered.add((r, c))
                    if (r, c) == anchor:
                        continue
                    target = rows[r].cells[c]
                    if target.value:
                        report.add(
                            SEVERITY_WARNING,
                            "covered_value_cleared",
                            f"value under merged region ({r},{c}) cleared",
                            loc,
                        )
                        target.value = ""
    table.merged_cells = merged


def _normalize_pages(result: ExtractionResult, report: ValidationReport) -> None:
    merged: dict[int, PageData] = {}
    for page in result.pages:
        number = page.page_number
        if number < 1:
            report.add(
                SEVERITY_ERROR,
                "invalid_page_number",
                f"page number {number} is invalid; clamped to 1",
                f"page {number}",
            )
            number = 1
            page.page_number = 1
        if number in merged:
            report.add(
                SEVERITY_WARNING,
                "duplicate_page",
                f"duplicate entry for page {number}; merged",
                f"page {number}",
            )
            merged[number].tables.extend(page.tables)
        else:
            merged[number] = page

    pages = [merged[key] for key in sorted(merged)]
    result.pages = pages

    for page in pages:
        for new_index, table in enumerate(page.tables):
            loc = f"page {page.page_number} table {table.table_index}"
            if table.page_number != page.page_number:
                report.add(
                    SEVERITY_WARNING,
                    "table_page_mismatch",
                    f"table page_number {table.page_number} corrected to {page.page_number}",
                    loc,
                )
                table.page_number = page.page_number
            if table.table_index != new_index:
                report.add(
                    SEVERITY_WARNING,
                    "table_index_remapped",
                    f"table index {table.table_index} remapped to {new_index}",
                    loc,
                )
                table.table_index = new_index
            _normalize_table(table, report, loc)

    if pages:
        max_page = max(p.page_number for p in pages)
        if max_page > result.page_count:
            report.add(
                SEVERITY_WARNING,
                "page_count_mismatch",
                f"reported page_count {result.page_count} < highest page {max_page}; fixed",
            )
            result.page_count = max_page
    elif result.page_count > 0:
        report.add(SEVERITY_WARNING, "no_pages_extracted", "extraction returned no pages")


def validate_and_normalize(
    result: ExtractionResult,
) -> tuple[ExtractionResult, ValidationReport]:
    """Validate and normalize ``result`` in place; returns (result, report)."""
    report = ValidationReport()
    _normalize_pages(result, report)
    return result, report


def is_numeric_text(value: str) -> bool:
    """True when the string contains digits (any script) plus separators only."""
    seen_digit = False
    for ch in value:
        if unicodedata.category(ch).startswith("Nd"):
            seen_digit = True
            continue
        if ch in ".,%+-–/()":
            continue
        if ch.isspace():
            continue
        return False
    return seen_digit


