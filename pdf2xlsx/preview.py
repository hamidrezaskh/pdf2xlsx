"""Preview helpers for the UI: build display frames from extracted tables.

Kept free of Streamlit so it can be unit-tested.  All transformations are
display-only — workbook contents are never affected.
"""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from .schemas import TableData


def unique_labels(headers: Sequence[str]) -> List[str]:
    """Make column labels unique for display.

    Real documents often repeat header names (or leave several columns
    unnamed).  pandas tolerates duplicate labels, but pyarrow (used by
    ``st.dataframe``) raises ``ValueError: Duplicate column names found``.
    The first occurrence keeps its text; later ones get a ``(n)`` suffix.
    """
    used: set[str] = set()
    result: List[str] = []
    for raw in headers:
        base = raw if isinstance(raw, str) else str(raw)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"({suffix})" if base == "" else f"{base} ({suffix})"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return result


def build_preview_frame(table: TableData) -> "pd.DataFrame":
    """Build a DataFrame preview of one table (unique column labels)."""
    headers, body = table.to_matrix()
    if headers and any(headers):
        return pd.DataFrame(body, columns=unique_labels(headers))
    return pd.DataFrame(body)


__all__ = ["unique_labels", "build_preview_frame"]
