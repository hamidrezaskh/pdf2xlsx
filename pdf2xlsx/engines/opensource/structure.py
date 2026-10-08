"""Reconstruct dynamic table structure from clustered text lines.

Rows, columns, headers, cell values, text direction and bounding boxes
are all inferred from the input document — nothing is hard-coded.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from ...pdf import Segment, WordBox
from ...schemas import BoundingBox, Cell, ColumnData, RowData, TableData
from ...validation import is_numeric_text
from .layout import (
    TextLine,
    build_lines,
    detect_grids,
    detect_whitespace_candidates,
    has_alpha,
    page_direction,
)

logger = logging.getLogger(__name__)

Band = Tuple[float, float]


def band_index(x_bands: Sequence[Band], x: float) -> int:
    """Column band for coordinate ``x`` (clamped to the table edges)."""
    for index, (start, end) in enumerate(x_bands):
        if start <= x < end:
            return index
    if x < x_bands[0][0]:
        return 0
    return len(x_bands) - 1


def _is_continuation(prev: TextLine, line: TextLine, x_bands: Sequence[Band]) -> bool:
    """True when ``line`` is a wrapped continuation of ``prev``'s row."""
    if not line.words or not prev.words:
        return False
    bands = {band_index(x_bands, w.cx) for w in line.words}
    if len(bands) != 1:
        return False
    band = next(iter(bands))
    prev_bands = {band_index(x_bands, w.cx) for w in prev.words}
    if band not in prev_bands:
        return False
    gap = line.y0 - prev.y1
    avg_height = (prev.height + line.height) / 2.0
    return gap <= 0.7 * avg_height


def group_lines_in_bands(
    lines: Sequence[TextLine], y_bands: Sequence[Band], x_bands: Sequence[Band]
) -> List[List[TextLine]]:
    """Assign lines to horizontal row bands (from ruling lines).

    A band with several text lines is split into separate rows, except
    when a line is a tight wrapped continuation inside one column.
    """
    rows: List[List[TextLine]] = []
    for y0, y1 in y_bands:
        band_lines = sorted(
            (ln for ln in lines if y0 <= ln.cy <= y1), key=lambda ln: ln.y0
        )
        if not band_lines:
            rows.append([])  # preserve genuinely empty ruled rows
            continue
        groups: List[List[TextLine]] = []
        for line in band_lines:
            if groups and _is_continuation(groups[-1][-1], line, x_bands):
                groups[-1].append(line)
            else:
                groups.append([line])
        rows.extend(groups)
    return rows


def merge_lines_into_rows(
    lines: Sequence[TextLine], x_bands: Sequence[Band]
) -> List[List[TextLine]]:
    """Group borderless text lines into rows (wrapped cells merged)."""
    groups: List[List[TextLine]] = []
    for line in sorted(lines, key=lambda ln: ln.y0):
        if groups and _is_continuation(groups[-1][-1], line, x_bands):
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _looks_like_header(rows: Sequence[RowData]) -> bool:
    """Conservative header test: text header above at least one numeric column."""
    if len(rows) < 2:
        return False
    first = rows[0]
    nonempty = [cell.value for cell in first.cells if cell.value.strip()]
    if not nonempty or not any(has_alpha(value) for value in nonempty):
        return False
    for index, cell in enumerate(first.cells):
        if not has_alpha(cell.value):
            continue
        below = [row.cells[index].value for row in rows[1:] if row.cells[index].value.strip()]
        if any(is_numeric_text(value) for value in below):
            return True
    return False


def build_table_data(
    *,
    page_number: int,
    table_index: int,
    x_bands: Sequence[Band],
    row_groups: Sequence[Sequence[TextLine]],
    space_w: float,
    space_h: float,
    source: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> TableData:
    """Assemble a TableData from column bands and row line groups."""
    all_lines = [line for group in row_groups for line in group]
    table = TableData(
        table_index=table_index,
        page_number=page_number,
        detection_source=source,
        text_direction=page_direction(all_lines) if all_lines else "ltr",
    )

    rows: List[RowData] = []
    confidences: List[float] = []
    table_y0: Optional[float] = None
    table_y1: Optional[float] = None

    for row_index, group in enumerate(row_groups):
        row_y0 = min((ln.y0 for ln in group), default=None)
        row_y1 = max((ln.y1 for ln in group), default=None)
        if row_y0 is not None:
            table_y0 = row_y0 if table_y0 is None else min(table_y0, row_y0)
            table_y1 = row_y1 if table_y1 is None else max(table_y1, row_y1)

        cells: List[Cell] = []
        for col_index, (bx0, bx1) in enumerate(x_bands):
            parts: Dict[int, List[WordBox]] = {}
            cell_confs: List[float] = []
            for order, line in enumerate(group):
                reverse = line.direction == "rtl"
                for word in sorted(line.words, key=lambda w: w.x0, reverse=reverse):
                    if col_index < len(x_bands) - 1:
                        inside = bx0 <= word.cx < bx1
                    else:
                        inside = bx0 <= word.cx <= bx1
                    if not inside:
                        continue
                    parts.setdefault(order, []).append(word)
                    if word.confidence is not None:
                        cell_confs.append(float(word.confidence))
            text = "\n".join(
                " ".join(w.text for w in parts[order]) for order in sorted(parts)
            )
            if cell_confs:
                confidence = sum(cell_confs) / len(cell_confs)
                confidences.extend(cell_confs)
            else:
                confidence = None
            bbox_cell = None
            if row_y0 is not None and row_y1 is not None:
                bbox_cell = BoundingBox(
                    x0=bx0 / space_w,
                    y0=row_y0 / space_h,
                    x1=bx1 / space_w,
                    y1=row_y1 / space_h,
                )
            cells.append(
                Cell(
                    row_index=row_index,
                    column_index=col_index,
                    value=text,
                    bbox=bbox_cell,
                    confidence=confidence,
                )
            )
        rows.append(RowData(index=row_index, cells=cells))

    table.rows = rows

    top = table_y0 if table_y0 is not None else (bbox[1] if bbox else 0.0)
    bottom = table_y1 if table_y1 is not None else (bbox[3] if bbox else 1.0)
    table.columns = [
        ColumnData(
            index=index,
            bbox=BoundingBox(
                x0=bx0 / space_w, y0=top / space_h, x1=bx1 / space_w, y1=bottom / space_h
            ),
        )
        for index, (bx0, bx1) in enumerate(x_bands)
    ]

    if rows and _looks_like_header(rows):
        rows[0].is_header = True

    if bbox is not None:
        table.bbox = BoundingBox(
            x0=bbox[0] / space_w,
            y0=bbox[1] / space_h,
            x1=bbox[2] / space_w,
            y1=bbox[3] / space_h,
        )
    elif all_lines:
        table.bbox = BoundingBox(
            x0=min(ln.x0 for ln in all_lines) / space_w,
            y0=min(ln.y0 for ln in all_lines) / space_h,
            x1=max(ln.x1 for ln in all_lines) / space_w,
            y1=max(ln.y1 for ln in all_lines) / space_h,
        )

    if confidences:
        table.confidence = sum(confidences) / len(confidences)
    return table


def detect_page_tables(
    *,
    page_number: int,
    words: Sequence[WordBox],
    h_segments: Sequence[Segment],
    v_segments: Sequence[Segment],
    space_w: float,
    space_h: float,
    grid_scale: Tuple[float, float] = (1.0, 1.0),
) -> List[TableData]:
    """Run both detectors on one page and return tables in reading order.

    ``grid_scale`` converts ruling-line coordinates (PDF points) into the
    coordinate space of ``words`` (points for native text, pixels for OCR).
    """
    lines = build_lines(words)
    if not lines:
        return []

    sx, sy = grid_scale
    tables: List[TableData] = []
    covered: List[Tuple[float, float, float, float]] = []

    # 1) tables defined by vector ruling lines -----------------------------
    for grid in detect_grids(h_segments, v_segments):
        x_edges = [x * sx for x in grid.x_edges]
        y_edges = [y * sy for y in grid.y_edges]
        gx0, gy0 = x_edges[0], y_edges[0]
        gx1, gy1 = x_edges[-1], y_edges[-1]
        inner = [ln for ln in lines if gx0 <= ln.cx <= gx1 and gy0 <= ln.cy <= gy1]
        if not inner:
            continue
        x_bands = list(zip(x_edges, x_edges[1:]))
        y_bands = list(zip(y_edges, y_edges[1:]))
        row_groups = group_lines_in_bands(inner, y_bands, x_bands)
        if not any(row_groups):
            continue
        table = build_table_data(
            page_number=page_number,
            table_index=len(tables),
            x_bands=x_bands,
            row_groups=row_groups,
            space_w=space_w,
            space_h=space_h,
            source="ruling_lines",
            bbox=(gx0, gy0, gx1, gy1),
        )
        tables.append(table)
        if table.bbox:
            covered.append((gx0, gy0, gx1, gy1))

    # 2) borderless tables from consistent column gaps ---------------------
    for candidate in detect_whitespace_candidates(lines):
        cx = (candidate.bbox[0] + candidate.bbox[2]) / 2.0
        cy = (candidate.bbox[1] + candidate.bbox[3]) / 2.0
        if any(x0 <= cx <= x1 and y0 <= cy <= y1 for (x0, y0, x1, y1) in covered):
            continue  # already covered by a ruled table
        row_groups = merge_lines_into_rows(candidate.lines, candidate.x_bands)
        if len(row_groups) < 2:
            continue
        table = build_table_data(
            page_number=page_number,
            table_index=len(tables),
            x_bands=candidate.x_bands,
            row_groups=row_groups,
            space_w=space_w,
            space_h=space_h,
            source="whitespace",
            bbox=candidate.bbox,
        )
        tables.append(table)
        covered.append(candidate.bbox)

    tables.sort(
        key=lambda t: (
            round(t.bbox.y0, 2) if t.bbox else 0.0,
            round(t.bbox.x0, 2) if t.bbox else 0.0,
        )
    )
    for index, table in enumerate(tables):
        table.table_index = index
    return tables


__all__ = [
    "band_index",
    "group_lines_in_bands",
    "merge_lines_into_rows",
    "build_table_data",
    "detect_page_tables",
]


