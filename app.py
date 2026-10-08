"""Streamlit UI for the PDF -> XLSX converter.

This layer handles UI concerns only: upload, mode selection, status,
preview and download.  All business logic lives in ``pdf2xlsx.pipeline``
and the engine modules.
"""

from __future__ import annotations

import logging

import streamlit as st

from pdf2xlsx.config import Settings
from pdf2xlsx.engines import OPENAI_MODE, engine_choices
from pdf2xlsx.exceptions import Pdf2XlsxError, friendly_message
from pdf2xlsx.pipeline import ConversionOutcome, convert_pdf
from pdf2xlsx.preview import build_preview_frame

logger = logging.getLogger(__name__)

st.set_page_config(page_title="PDF → XLSX", page_icon="📄", layout="wide")


def _mode_options() -> tuple[list[str], dict[str, str]]:
    """Radio labels + label→engine-name map, straight from the registry.

    Newly registered engines show up here automatically — the UI never
    hard-codes the list of processors.
    """
    labels: list[str] = []
    label_to_name: dict[str, str] = {}
    for name, label in engine_choices():
        unique = label
        counter = 2
        while unique in label_to_name:  # de-duplicate identical labels
            unique = f"{label} ({counter})"
            counter += 1
        labels.append(unique)
        label_to_name[unique] = name
    return labels, label_to_name


@st.cache_resource
def get_settings() -> Settings:
    """Settings are read from the environment once per app run."""
    return Settings.from_env()


def _run_conversion(uploaded, mode: str, settings: Settings) -> None:
    st.session_state.pop("outcome", None)
    data = uploaded.getvalue()
    with st.status("Processing...", expanded=True) as status:

        def on_progress(message: str) -> None:
            st.write(message)

        try:
            outcome = convert_pdf(
                data,
                mode,
                settings,
                filename=uploaded.name,
                on_progress=on_progress,
            )
        except Pdf2XlsxError as exc:
            status.update(label="Failed", state="error", expanded=True)
            logger.error("conversion failed: %s (%s)", exc.message, exc.detail)
            st.error(friendly_message(exc))
            return
        except Exception as exc:  # pragma: no cover - safety net
            status.update(label="Failed", state="error", expanded=True)
            logger.exception("unexpected conversion failure")
            st.error(friendly_message(exc))
            return
        status.update(label="Done", state="complete")
    st.session_state["outcome"] = outcome


def _render_preview_table(table) -> None:
    # unique display labels: pyarrow rejects duplicate DataFrame columns
    st.dataframe(build_preview_frame(table), use_container_width=True)


def _render_outcome(outcome: ConversionOutcome) -> None:
    result = outcome.result

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pages", result.page_count)
    col2.metric("Tables", result.table_count)
    mode_label = result.processing_mode
    if result.model_name:
        mode_label = f"{mode_label} · {result.model_name}"
    col3.metric("Mode", mode_label)
    col4.metric("Duration", f"{outcome.duration_seconds:.1f}s")

    if outcome.validation.issues:
        with st.expander(f"Validation — {outcome.validation.summary}", expanded=False):
            for issue in outcome.validation.issues:
                text = f"**{issue.code}**"
                if issue.location:
                    text += f" ({issue.location})"
                text += f": {issue.message}"
                if issue.severity == "error":
                    st.error(text)
                else:
                    st.warning(text)

    if result.warnings:
        st.warning("\n\n".join(result.warnings))

    st.subheader("Extraction preview")
    if result.table_count == 0:
        st.info("No tables were detected in this document.")
    for page, table in result.iter_tables():
        label = (
            f"Page {page.page_number} · Table {table.table_index + 1} "
            f"· {table.n_rows} rows × {table.n_columns} cols"
            f" · {table.text_direction}"
        )
        with st.expander(label, expanded=(result.table_count == 1)):
            _render_preview_table(table)

    st.download_button(
        "Download XLSX",
        data=outcome.xlsx_bytes,
        file_name=outcome.xlsx_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    if outcome.debug_dir:
        st.caption(f"Debug artifacts: `{outcome.debug_dir}`")


def main() -> None:
    settings = get_settings()

    st.title("PDF → XLSX")
    st.caption(
        "Dynamic table extraction: tables, rows, columns and headers are "
        "detected from the document — nothing is hard-coded."
    )

    with st.sidebar:
        st.header("Configuration")
        st.code(
            "\n".join(
                [
                    f"OPENAI_BASE_URL={settings.openai_base_url or '(default: api.openai.com)'}",
                    f"OPENAI_MODEL={settings.openai_model}",
                    f"OCR_MODEL={settings.ocr_model}",
                    f"OCR_LANGUAGES={','.join(settings.ocr_languages)}",
                    f"DEVICE={settings.device}",
                    f"DEBUG={settings.debug}",
                ]
            ),
            language="text",
        )
        st.caption("Change values via environment variables (see `.env.example`).")
        labels, label_to_name = _mode_options()
        st.markdown(
            f"**Engines:** {', '.join(f'{name} ({label})' for name, label in engine_choices())}"
        )
        if (
            OPENAI_MODE in label_to_name.values()
            and not settings.openai_api_key
            and not settings.openai_base_url
        ):
            st.warning(
                "OPENAI_API_KEY is not set — OpenAI-compatible mode needs a key "
                "or an OPENAI_BASE_URL."
            )

    uploaded = st.file_uploader("PDF file", type=["pdf"])
    mode_label = st.radio("Processing mode", labels, horizontal=True, index=0)

    if st.button(
        "Convert to XLSX",
        type="primary",
        disabled=uploaded is None,
        use_container_width=False,
    ):
        _run_conversion(uploaded, label_to_name[mode_label], settings)

    outcome: ConversionOutcome | None = st.session_state.get("outcome")
    if outcome is not None:
        st.divider()
        _render_outcome(outcome)


if __name__ == "__main__":
    main()
