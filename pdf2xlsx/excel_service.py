"""Excel service: JSON/records -> XLSX with RTL, headers and auto-width.

Responsibilities:

* convert :class:`~pdf2xlsx.schemas.ExtractionResult` to XLSX bytes
  (delegates to :mod:`pdf2xlsx.xlsx`, preserving one-sheet-per-table,
  merged cells, header styling, RTL sheets and column auto-sizing)
* convert plain JSON records (``list[dict]``) to XLSX while **preserving
  field order**, building headers from the schema, enabling RTL and
  tuning column widths
* helpers for sheet-name sanitising shared by both paths
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .schemas import ExtractionResult
from .xlsx import generate_xlsx

logger = logging.getLogger(__name__)

_HEADER_FONT = Font(bold=True)
_HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
_MAX_COL_WIDTH = 60


# ---------------------------------------------------------------------------
# ExtractionResult path (tables already detected)
# ---------------------------------------------------------------------------


def result_to_xlsx(result: ExtractionResult, *, style: bool = True) -> bytes:
    """Serialize a normalized extraction result into XLSX bytes."""
    return generate_xlsx(result, style=style)


# ---------------------------------------------------------------------------
# JSON records path (user-schema output)
# ---------------------------------------------------------------------------


def _display_width(value: str) -> int:
    width = 0
    for ch in value:
        width += 2 if ord(ch) > 0x0600 else 1  # rough RTL/CJK widening
    return width


def _is_rtl_text(text: str) -> bool:
    from .textutil import first_strong_direction

    return first_strong_direction(text) == "rtl"


def records_to_xlsx(
    records: Sequence[Mapping[str, Any]],
    *,
    field_order: Optional[Sequence[str]] = None,
    sheet_name: str = "Sheet1",
    rtl: Optional[bool] = None,
    header: bool = True,
    style: bool = True,
    max_col_width: int = _MAX_COL_WIDTH,
) -> bytes:
    """Convert JSON records to XLSX bytes.

    * ``field_order`` — explicit column order; defaults to first-seen key
      order (insertion order preserved, never sorted alphabetically).
    * ``rtl`` — None = auto-detect from content; True/False forces direction.
    * column widths are auto-sized from content (capped at ``max_col_width``).
    """
    headers: List[str] = list(field_order) if field_order else []
    if not headers:
        for rec in records:
            for key in rec.keys():
                if key not in headers:
                    headers.append(str(key))

    workbook = Workbook()
    ws = workbook.active
    ws.title = sheet_name[:31] or "Sheet1"

    # direction
    if rtl is None:
        sample = " ".join(headers + [str(v) for r in records[:5] for v in r.values()])
        rtl = _is_rtl_text(sample)
    if rtl:
        ws.sheet_view.rightToLeft = True

    widths: List[int] = [max(8, _display_width(h) + 2) for h in headers]

    start_row = 1
    if header and headers:
        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=c, value=h)
            if style:
                cell.font = _HEADER_FONT
                cell.fill = _HEADER_FILL
                cell.alignment = Alignment(horizontal="center", vertical="center")
        start_row = 2
        ws.freeze_panes = "A2"

    for r, rec in enumerate(records, start=start_row):
        for c, h in enumerate(headers, start=1):
            value = rec.get(h, "")
            if value == "":
                value = None
            cell = ws.cell(row=r, column=c, value=value)
            if style and value is not None:
                cell.alignment = Alignment(
                    vertical="center", wrap_text="\n" in str(value)
                )
            if value is not None:
                for part in str(value).split("\n"):
                    widths[c - 1] = max(widths[c - 1], _display_width(part) + 2)

    for c in range(len(headers)):
        ws.column_dimensions[get_column_letter(c + 1)].width = min(
            widths[c], max_col_width
        )

    buffer = BytesIO()
    workbook.save(buffer)
    data = buffer.getvalue()
    logger.info("records XLSX: %d row(s), %d col(s), %d byte(s)", len(records), len(headers), len(data))
    return data


def multi_sheet_records_to_xlsx(
    sheets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field_orders: Optional[Mapping[str, Sequence[str]]] = None,
    rtl: Optional[bool] = None,
    style: bool = True,
) -> bytes:
    """Multiple named record sets -> one XLSX with one sheet per key."""
    workbook = Workbook()
    workbook.remove(workbook.active)
    used: set[str] = set()
    field_orders = field_orders or {}

    for raw_name, records in sheets.items():
        name = (raw_name or "Sheet")[:31] or "Sheet"
        candidate, i = name, 2
        while candidate.lower() in used:
            candidate = f"{name[:28]}_{i}"
            i += 1
        used.add(candidate.lower())
        ws = workbook.create_sheet(title=candidate)

        headers: List[str] = list(field_orders.get(raw_name, []))
        if not headers:
            for rec in records:
                for key in rec.keys():
                    if key not in headers:
                        headers.append(str(key))

        if rtl:
            ws.sheet_view.rightToLeft = True

        widths = [max(8, _display_width(h) + 2) for h in headers]
        if headers:
            for c, h in enumerate(headers, start=1):
                cell = ws.cell(row=1, column=c, value=h)
                if style:
                    cell.font = _HEADER_FONT
                    cell.fill = _HEADER_FILL
                    cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.freeze_panes = "A2"
        for r, rec in enumerate(records, start=2 if headers else 1):
            for c, h in enumerate(headers, start=1):
                value = rec.get(h, "")
                ws.cell(row=r, column=c, value=value if value != "" else None)
                if value not in ("", None):
                    for part in str(value).split("\n"):
                        widths[c - 1] = max(widths[c - 1], _display_width(part) + 2)
        for c in range(len(headers)):
            ws.column_dimensions[get_column_letter(c + 1)].width = min(
                widths[c], _MAX_COL_WIDTH
            )

    if not workbook.sheetnames:
        workbook.create_sheet(title="NoData")
    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


__all__ = [
    "result_to_xlsx",
    "records_to_xlsx",
    "multi_sheet_records_to_xlsx",
]
