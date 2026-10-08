"""Shared helpers for LLM-backed extraction engines (prompt + JSON parsing).

LLM processors use the same generic schema description — no
domain-specific assumptions about columns, layouts or document types.
"""

from __future__ import annotations

import json
import re

from .exceptions import InvalidStructuredResponseError
from .schemas import ExtractionPayload

# Generic prompt: no domain-specific columns, layouts or assumptions.
PROMPT_TEMPLATE = """You are a table-structure extraction engine for documents.

Analyze __SUBJECT__ and extract EVERY table it contains.

Return ONLY a JSON object with this exact structure:
{
  "pages": [
    {
      "page_number": <1-based page number>,
      "tables": [
        {
          "table_index": <0-based, ordered top-to-bottom on the page>,
          "bbox": {"x0": <0..1>, "y0": <0..1>, "x1": <0..1>, "y1": <0..1>},
          "text_direction": "rtl" or "ltr",
          "confidence": <0..1>,
          "columns": [
            {
              "index": <0-based, ordered left-to-right>,
              "header": <column name or null>,
              "bbox": {
                "x0": <0..1>,
                "y0": <0..1>,
                "x1": <0..1>,
                "y1": <0..1>
              } or null
            }
          ],
          "rows": [
            {
              "index": <0-based, ordered top-to-bottom>,
              "is_header": <true if this row is the header row>,
              "cells": [
                {
                  "row_index": <same value as the containing row index>,
                  "column_index": <0-based, left-to-right>,
                  "value": "<exact cell text>",
                  "rowspan": <integer >= 1>,
                  "colspan": <integer >= 1>,
                  "confidence": <0..1 or null>,
                  "bbox": {
                    "x0": <0..1>,
                    "y0": <0..1>,
                    "x1": <0..1>,
                    "y1": <0..1>
                  } or null
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

Rules:
- __SCOPE__
- Infer tables, rows, columns, column names, and cell boundaries dynamically
  from the document.
- NEVER assume a fixed number of columns, rows, header names, or any
  particular table layout.
- Detect the actual table structure from the document itself.
- bbox coordinates are normalized page fractions (0..1), with origin at
  the top-left corner.
- row_index increases from top to bottom.
- column_index increases from left to right, even for RTL tables.
- Mark the actual header row with "is_header": true.
- Copy each header cell's text exactly into the matching column's "header".
- Preserve cell text EXACTLY as written in the document.
- Do NOT translate, normalize, rewrite, correct, or interpret cell text.
- Do NOT convert digits between Persian, Arabic, and Latin forms.
- Preserve Persian, Arabic, and Latin characters exactly as they appear.
- Preserve mixed RTL/LTR content exactly, including numbers, symbols,
  punctuation, units, and spacing whenever possible.
- Empty cells MUST still be included with "value": "".
- For merged cells, put the value only in the top-left cell, set the
  appropriate rowspan/colspan, and OMIT all cells covered by the merge.
- NEVER invent, infer, complete, or fabricate content that is not visible
  in the document.
- If a cell is partially or completely unreadable, preserve the readable
  text and use an empty string for the unreadable portion rather than
  guessing.
- If a table contains separate Width and Height information, they MUST be
  represented as separate columns.
- NEVER combine Width and Height into a single column when the document
  represents them as separate fields.
- If a dimension is written as a combined expression such as
  "2502 * 1280", preserve the original cell text exactly unless the
  document itself clearly represents Width and Height as separate columns.
- Detect separate columns based on the actual visual structure of the table,
  not based on expected or predefined column names.
- If multiple tables exist on the same page, detect and extract each table
  separately.
- table_index must reflect the top-to-bottom order of tables on each page.
- rows must preserve their visual top-to-bottom order.
- Do not create columns or rows that do not exist in the document.
- Do not omit visible rows, columns, or cells.
- Return valid JSON only. Do not include Markdown, explanations,
  comments, or additional text outside the JSON object.
"""


def build_extraction_prompt(total_pages: int, page_number: int | None = None) -> str:
    """Build the generic extraction prompt.

    ``page_number=None`` targets a whole PDF; otherwise the prompt
    targets a single rendered page image (vision endpoints).
    """
    if page_number is None:
        subject = f"the attached PDF (total pages: {total_pages})"
        scope = (
            f"Include every page that contains tables; use page_number values 1..{total_pages}. "
            'Pages without tables may be omitted or have "tables": [].'
        )
    else:
        subject = f"the attached page image (page {page_number} of {total_pages})"
        scope = (
            f"Return ONLY tables visible on page {page_number} and set page_number to {page_number}. "
            'If there are no tables on this page, return {"pages": []}.'
        )
    return PROMPT_TEMPLATE.replace("__SUBJECT__", subject).replace("__SCOPE__", scope)


def strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` fences if the model added them anyway."""
    stripped = text.strip()
    match = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", stripped, re.DOTALL)
    if match:
        return match.group(1)
    return stripped


def payload_from_text(text: str) -> ExtractionPayload:
    """Parse free-form model text output into the structured payload."""
    if not text or not str(text).strip():
        raise InvalidStructuredResponseError(
            "The model returned an empty response; no table data could be parsed."
        )
    try:
        data = json.loads(strip_code_fences(str(text)))
    except json.JSONDecodeError as exc:
        raise InvalidStructuredResponseError(
            "The model returned a response that is not valid JSON.",
            detail=f"{exc}",
        ) from exc
    try:
        return ExtractionPayload.model_validate(data)
    except Exception as exc:
        raise InvalidStructuredResponseError(
            "The model response did not match the expected table structure.",
            detail=str(exc),
        ) from exc


__all__ = [
    "PROMPT_TEMPLATE",
    "build_extraction_prompt",
    "strip_code_fences",
    "payload_from_text",
]
