"""Normalized structured data model.

Every extraction engine (OpenAI-compatible API, open-source OCR, future
engines) produces exactly this representation.  The XLSX generation
layer consumes only this representation and contains no engine-specific
logic.

Nothing here is document- or domain-specific: columns, rows, headers,
dimensions and coordinates are always inferred from the input document.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Page-relative bounding box, normalized to 0..1 (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float


class Cell(BaseModel):
    """A single table cell."""

    row_index: int
    column_index: int
    value: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: Optional[float] = None
    rowspan: int = 1
    colspan: int = 1


class ColumnData(BaseModel):
    """A detected column. ``header`` is the detected column name, if any."""

    index: int
    header: Optional[str] = None
    bbox: Optional[BoundingBox] = None


class RowData(BaseModel):
    """A detected row; ``is_header`` marks header rows."""

    index: int
    is_header: bool = False
    cells: List[Cell] = Field(default_factory=list)


class MergedCell(BaseModel):
    """A merged cell region anchored at (row_index, column_index)."""

    row_index: int
    column_index: int
    rowspan: int
    colspan: int
    value: str = ""
    bbox: Optional[BoundingBox] = None


class TableData(BaseModel):
    """One detected table on one page.

    ``row_index`` increases top to bottom; ``column_index`` increases left
    to right regardless of text direction.  ``text_direction`` carries the
    dominant writing direction (``rtl`` for Persian/Arabic, ``ltr`` else).
    """

    table_index: int = 0
    page_number: int = 1
    bbox: Optional[BoundingBox] = None
    columns: List[ColumnData] = Field(default_factory=list)
    rows: List[RowData] = Field(default_factory=list)
    merged_cells: List[MergedCell] = Field(default_factory=list)
    confidence: Optional[float] = None
    text_direction: Literal["ltr", "rtl"] = "ltr"
    detection_source: Optional[str] = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def header_labels(self) -> List[str]:
        return [col.header or "" for col in self.columns]

    def to_matrix(self, *, include_header_row: bool = True) -> tuple[List[str], List[List[str]]]:
        """Return (headers, body rows) for previews / display."""
        headers = self.header_labels()
        body: List[List[str]] = []
        for row in self.rows:
            if row.is_header and include_header_row and any(headers):
                continue  # already represented by the column headers
            body.append([cell.value for cell in row.cells])
        return headers, body


class PageData(BaseModel):
    """All tables detected on a single page (page numbers are 1-based)."""

    page_number: int
    tables: List[TableData] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Normalized result shared by every extraction engine."""

    source_filename: str = ""
    page_count: int = 0
    processing_mode: str = ""
    model_name: Optional[str] = None
    pages: List[PageData] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def iter_tables(self):
        """Yield (PageData, TableData) for every table in document order."""
        for page in self.pages:
            for table in page.tables:
                yield page, table

    @property
    def table_count(self) -> int:
        return sum(len(page.tables) for page in self.pages)


class ExtractionPayload(BaseModel):
    """Wire format requested from generative models (structured output)."""

    pages: List[PageData] = Field(default_factory=list)
