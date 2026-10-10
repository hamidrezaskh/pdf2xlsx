"""Schema service: user-defined schemas + validation + defaults.

Two layers:

1. **Fixed engine schema** — :mod:`pdf2xlsx.schemas` /
   :mod:`pdf2xlsx.validation` (tables, rows, cells, bboxes). Re-exported
   here so callers only depend on this service.
2. **User schema** — the caller (UI/API) declares which fields it wants
   from each table row: required/optional, default values, type coercion.
   This is what enables "ساخت Schema توسط کاربر" without touching the
   extraction engines.

Example::

    from pdf2xlsx.schema_service import FieldSpec, UserTableSchema

    schema = UserTableSchema(fields=[
        FieldSpec(name="invoice_no", required=True),
        FieldSpec(name="amount", required=False, default="0"),
    ])
    records = schema.apply(table)   # list[dict] in declared field order
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .schemas import (
    BoundingBox,
    Cell,
    ColumnData,
    ExtractionPayload,
    ExtractionResult,
    MergedCell,
    PageData,
    RowData,
    TableData,
)
from .validation import (
    ValidationIssue,
    ValidationReport,
    validate_and_normalize,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# fixed engine schema (re-exports + helpers)
# ---------------------------------------------------------------------------


def validate_result(result: ExtractionResult) -> tuple[ExtractionResult, ValidationReport]:
    """Validate + normalize engine output (in place)."""
    return validate_and_normalize(result)


def table_headers(table: TableData) -> List[str]:
    return table.header_labels()


# ---------------------------------------------------------------------------
# user-defined schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One user-declared output field.

    * ``name`` — output column name (matched against detected headers,
      case-insensitive; falls back to positional mapping).
    * ``required`` — missing values raise a validation issue.
    * ``default`` — value used when the cell is empty and not required.
    * ``aliases`` — alternative header spellings accepted for this field.
    """

    name: str
    required: bool = False
    default: Any = ""
    aliases: tuple[str, ...] = ()

    def matches(self, header: str) -> bool:
        h = (header or "").strip().lower()
        if h == self.name.strip().lower():
            return True
        return any(h == a.strip().lower() for a in self.aliases)


@dataclass
class UserTableSchema:
    """Ordered user schema applied to every extracted table."""

    fields: List[FieldSpec] = field(default_factory=list)
    #: when True, extra detected columns not in ``fields`` are kept as-is.
    keep_extra_columns: bool = True
    #: when True, header matching is positional (field[i] <- column[i]).
    positional_fallback: bool = True

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names in schema: {names}")

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def column_mapping(self, headers: List[str]) -> Dict[int, Optional[int]]:
        """Map detected column index -> field index (None = extra column)."""
        mapping: Dict[int, Optional[int]] = {}
        used_fields: set[int] = set()
        for col_idx, header in enumerate(headers):
            matched: Optional[int] = None
            for fi, spec in enumerate(self.fields):
                if fi in used_fields:
                    continue
                if header.strip() and spec.matches(header):
                    matched = fi
                    used_fields.add(fi)
                    break
            mapping[col_idx] = matched
        # positional fallback for unmatched fields/columns
        if self.positional_fallback:
            unmatched_cols = [c for c, f in mapping.items() if f is None]
            unmatched_fields = [fi for fi in range(len(self.fields)) if fi not in used_fields]
            for col_idx, fi in zip(unmatched_cols, unmatched_fields):
                mapping[col_idx] = fi
        return mapping

    def apply_to_table(self, table: TableData) -> tuple[List[Dict[str, Any]], List[str]]:
        """Convert a table's body rows to ordered dicts; returns (records, issues)."""
        headers = table.header_labels()
        mapping = self.column_mapping(headers)
        _, body = table.to_matrix(include_header_row=True)
        records: List[Dict[str, Any]] = []
        issues: List[str] = []
        for r, row_values in enumerate(body):
            record: Dict[str, Any] = {}
            for col_idx, spec in enumerate(self.fields):
                # find detected column mapped to this field
                src = next((c for c, f in mapping.items() if f == col_idx), None)
                value: Any = ""
                if src is not None and src < len(row_values):
                    value = row_values[src]
                if (value is None or str(value).strip() == "") and spec.default != "":
                    value = spec.default
                elif value is None:
                    value = ""
                if spec.required and str(value).strip() == "":
                    issues.append(f"row {r}: required field '{spec.name}' is empty")
                record[spec.name] = value
            if self.keep_extra_columns:
                for col_idx, val in enumerate(row_values):
                    if mapping.get(col_idx) is None:
                        header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                        key = header or f"col_{col_idx}"
                        # avoid clobbering declared fields
                        if key not in record:
                            record[key] = val
            records.append(record)
        return records, issues

    def apply_to_result(self, result: ExtractionResult) -> Dict[str, List[Dict[str, Any]]]:
        """Apply to every table; keys are ``P{page}T{index}`` sheet names."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for page, table in result.iter_tables():
            key = f"P{page.page_number}T{table.table_index + 1}"
            records, _ = self.apply_to_table(table)
            out[key] = records
        return out


class SchemaRequest(BaseModel):
    """Wire model for a user-submitted schema (e.g. from the UI/API)."""

    fields: List[Dict[str, Any]]

    def to_user_schema(self) -> UserTableSchema:
        specs = []
        for item in self.fields:
            specs.append(
                FieldSpec(
                    name=str(item.get("name", "")),
                    required=bool(item.get("required", False)),
                    default=item.get("default", ""),
                    aliases=tuple(item.get("aliases", []) or []),
                )
            )
        return UserTableSchema(fields=specs)


def build_default_schema(tables: List[TableData]) -> UserTableSchema:
    """Infer a user schema from detected headers (all optional, no defaults)."""
    seen: List[str] = []
    for table in tables:
        for header in table.header_labels():
            name = (header or "").strip()
            if name and name not in seen:
                seen.append(name)
    if not seen and tables:
        width = max((t.n_columns for t in tables), default=0)
        seen = [f"col_{i}" for i in range(width)]
    return UserTableSchema(fields=[FieldSpec(name=n) for n in seen])


__all__ = [
    # fixed schema re-exports
    "BoundingBox",
    "Cell",
    "ColumnData",
    "RowData",
    "MergedCell",
    "TableData",
    "PageData",
    "ExtractionResult",
    "ExtractionPayload",
    "ValidationIssue",
    "ValidationReport",
    "validate_result",
    "table_headers",
    # user schema
    "FieldSpec",
    "UserTableSchema",
    "SchemaRequest",
    "build_default_schema",
]
