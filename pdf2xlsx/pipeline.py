"""End-to-end conversion orchestration.

Pipeline: PDF -> OCR/LLM extraction -> schema validation/normalization ->
XLSX generation, with logging, progress callbacks and debug artifacts.

This module orchestrates the modular services only:

* :mod:`pdf2xlsx.ocr_service`   — PDF/image input, OCR, bboxes, confidence
* :mod:`pdf2xlsx.llm_service`   — Gemini/LLM structured extraction
* :mod:`pdf2xlsx.schema_service` — schema definition + validation/defaults
* :mod:`pdf2xlsx.excel_service` — JSON -> XLSX (order, header, RTL, widths)

Engines (``pdf2xlsx.engines``) implement the extraction strategies and are
selected via the registry; this module never contains engine logic itself.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import Settings
from .debug import DebugWriter
from .engines import ExtractionEngine, get_engine, normalize_mode
from .exceptions import ExtractionError
from .excel_service import result_to_xlsx
from .logging_config import setup_logging
from .ocr_service import load_pdf_input
from .pdf import PdfSource, load_pdf
from .schema_service import validate_result
from .schemas import ExtractionResult
from .validation import ValidationReport

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


@dataclass
class ConversionOutcome:
    """Everything the UI needs after a successful conversion."""

    result: ExtractionResult
    validation: ValidationReport
    xlsx_bytes: bytes
    duration_seconds: float
    debug_dir: Optional[Path] = None

    @property
    def xlsx_filename(self) -> str:
        return xlsx_filename_for(self.result.source_filename)


def xlsx_filename_for(source_name: str) -> str:
    stem = Path(source_name or "document").stem or "document"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .") or "document"
    return f"{safe}.xlsx"


def convert_pdf(
    source: PdfSource,
    mode: str,
    settings: Optional[Settings] = None,
    *,
    filename: Optional[str] = None,
    engine: Optional[ExtractionEngine] = None,
    debug: Optional[DebugWriter] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> ConversionOutcome:
    """Run the full pipeline and return the outcome.

    ``engine`` may be injected (tests); otherwise the engine is created
    from the registry based on ``mode``.
    """
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level)

    started = time.perf_counter()
    mode_key = normalize_mode(mode)

    def emit(message: str) -> None:
        logger.info(message)
        if on_progress is not None:
            try:
                on_progress(message)
            except Exception:  # pragma: no cover - UI callbacks must not break work
                logger.debug("progress callback failed", exc_info=True)

    emit(f"Processing mode: {mode_key}")
    if debug is None and settings.debug:
        debug = DebugWriter(settings.debug_dir)

    pdf = load_pdf_input(source, source_name=filename)
    try:
        emit(f"PDF loaded: {pdf.page_count} page(s)")
        active = engine or get_engine(mode_key, settings)
        model = active.model_name
        emit(
            f"Extracting tables with '{active.name}'"
            + (f" (model: {model})" if model else "")
            + "..."
        )
        result = active.extract(pdf, debug=debug)
        emit(f"Extraction finished: {result.table_count} table(s) detected")

        result, report = validate_result(result)
        emit(f"Validation finished: {report.summary}")
        for issue in report.issues:
            logger.log(
                logging.ERROR if issue.severity == "error" else logging.WARNING,
                "validation: %s (%s)",
                issue.message,
                issue.location or "-",
            )

        if result.table_count == 0 and result.warnings:
            # something went wrong (OCR/config); surface it instead of an
            # silently empty workbook
            summary = " ".join(result.warnings[:3])
            raise ExtractionError(summary)

        emit("Generating XLSX...")
        xlsx_bytes = result_to_xlsx(result)
        duration = time.perf_counter() - started

        debug_dir: Optional[Path] = None
        if debug is not None:
            debug.save_json("structured_result.json", result)
            debug.save_xlsx("result.xlsx", xlsx_bytes)
            debug_dir = debug.dir
            emit(f"Debug artifacts saved to {debug_dir}")

        emit(f"Done in {duration:.1f}s")
        return ConversionOutcome(
            result=result,
            validation=report,
            xlsx_bytes=xlsx_bytes,
            duration_seconds=duration,
            debug_dir=debug_dir,
        )
    finally:
        pdf.close()
        logger.info(
            "conversion finished: mode=%s elapsed=%.2fs",
            mode_key,
            time.perf_counter() - started,
        )


__all__ = ["ConversionOutcome", "convert_pdf", "xlsx_filename_for"]
