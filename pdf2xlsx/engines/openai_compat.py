"""OpenAI-compatible extraction engine.

Works with any endpoint that speaks the OpenAI Chat Completions API:
OpenAI, OpenRouter, Azure-compatible gateways, Ollama, vLLM, llama.cpp
server, LM Studio, ...

Strategy: each PDF page is rendered to an image and sent as a vision
message together with the shared generic extraction prompt; the model
returns JSON matching :class:`~pdf2xlsx.schemas.ExtractionPayload`.
JSON mode (``response_format``) is used when the server supports it and
transparently retried without it otherwise.

The API key / base URL / model are read from the environment only.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from ..config import Settings
from ..debug import DebugWriter
from ..exceptions import (
    ConfigurationError,
    InvalidStructuredResponseError,
    OpenAiApiError,
    Pdf2XlsxError,
)
from ..llm import build_extraction_prompt, payload_from_text
from ..pdf import PdfDocument
from ..schemas import ExtractionResult, PageData
from . import OPENAI_MODE, register_engine
from .base import ExtractionEngine

logger = logging.getLogger(__name__)


def _translate_api_error(exc: Exception) -> OpenAiApiError:
    """Map SDK/network exceptions to user-friendly, secret-free messages."""
    message = str(exc)
    lowered = message.lower()
    cls_name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    if (
        isinstance(exc, (ConnectionError, TimeoutError))
        or "connection" in cls_name.lower()
        or "timeout" in cls_name.lower()
        or any(t in lowered for t in ("timed out", "timeout", "connection", "unreachable"))
    ):
        return OpenAiApiError(
            "Network error while contacting the OpenAI-compatible API. "
            "Check OPENAI_BASE_URL and your connectivity, then try again.",
            detail=f"{cls_name}: {message}",
        )
    if (
        status in (401, 403)
        or "authentication" in cls_name.lower()
        or any(t in lowered for t in ("unauthorized", "invalid api key", "invalid_api_key", "permission"))
    ):
        return OpenAiApiError(
            "The API key was rejected. Check OPENAI_API_KEY (or your endpoint's auth) and try again.",
            detail=f"{cls_name}: {message}",
        )
    if status == 429 or any(t in lowered for t in ("rate limit", "quota", "too many requests")):
        return OpenAiApiError(
            "The API rate limit or quota was exceeded. Please retry later.",
            detail=f"{cls_name}: {message}",
        )
    return OpenAiApiError(
        "The OpenAI-compatible API request failed. Please try again.",
        detail=f"{cls_name}: {message}",
    )


def _is_response_format_unsupported(exc: Exception) -> bool:
    """True when the server rejected ``response_format`` (JSON mode)."""
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status == 400 and (
        "response_format" in text
        or "json_object" in text
        or "json mode" in text
        or ("json" in text and "not supported" in text)
    )


@register_engine
class OpenAICompatibleEngine(ExtractionEngine):
    """Vision-based extraction through an OpenAI-compatible endpoint."""

    name = OPENAI_MODE
    description = "OpenAI-Compatible API"

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        super().__init__(settings)
        self._client = client  # injectable for tests

    @property
    def model_name(self) -> Optional[str]:
        return self.settings.openai_model

    # -- client -----------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self.settings.openai_api_key
        base_url = self.settings.openai_base_url
        if not key and not base_url:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Add it to your environment (copy "
                ".env.example to .env), or set OPENAI_BASE_URL for a local/"
                "compatible server (e.g. Ollama) that needs no key."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ConfigurationError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc
        try:
            self._client = OpenAI(
                api_key=key or "not-needed",  # local servers ignore the key
                base_url=base_url or None,
            )
        except Exception as exc:
            raise _translate_api_error(exc) from exc
        return self._client

    # -- extraction -------------------------------------------------------
    def extract(
        self, pdf: PdfDocument, *, debug: Optional[DebugWriter] = None
    ) -> ExtractionResult:
        client = self._get_client()
        result = ExtractionResult(
            source_filename=pdf.source_name,
            page_count=pdf.page_count,
            processing_mode=self.name,
            model_name=self.settings.openai_model,
        )
        logger.info(
            "OpenAI-compatible request: model=%s pages=%d base_url=%s",
            self.settings.openai_model,
            pdf.page_count,
            self.settings.openai_base_url or "(default: api.openai.com)",
        )
        for page_number in range(1, pdf.page_count + 1):
            try:
                page_data = self._extract_page(client, pdf, page_number, debug)
            except Pdf2XlsxError as exc:
                # one bad page must not abort the whole document
                logger.warning("page %d failed: %s", page_number, exc.message)
                result.warnings.append(f"Page {page_number}: {exc.message}")
                page_data = PageData(page_number=page_number, tables=[])
            result.pages.append(page_data)
        logger.info(
            "OpenAI-compatible extraction finished: %d page(s), %d table(s), %d warning(s)",
            len(result.pages),
            result.table_count,
            len(result.warnings),
        )
        return result

    def _extract_page(
        self,
        client: Any,
        pdf: PdfDocument,
        page_number: int,
        debug: Optional[DebugWriter],
    ) -> PageData:
        png = pdf.render_png(page_number, self.settings.ocr_dpi)
        if debug is not None:
            debug.save_page_image(page_number, png)

        prompt = build_extraction_prompt(pdf.page_count, page_number=page_number)
        encoded = base64.b64encode(png).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]
        text = self._create(client, content)
        if debug is not None:
            debug.save_text(f"openai_response_page_{page_number:03d}.txt", text)

        payload = payload_from_text(text)
        tables = [table for page in payload.pages for table in page.tables]
        # trust our own page numbering over the model's
        for table in tables:
            table.page_number = page_number
        return PageData(page_number=page_number, tables=tables)

    def _create(self, client: Any, content: list) -> str:
        """Chat-completions call with JSON mode + graceful fallback."""
        kwargs: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": float(self.settings.openai_temperature),
            "response_format": {"type": "json_object"},
        }
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not _is_response_format_unsupported(exc):
                raise _translate_api_error(exc) from exc
            logger.info("server rejected response_format; retrying without JSON mode")
            kwargs.pop("response_format", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as retry_exc:
                raise _translate_api_error(retry_exc) from retry_exc

        try:
            text = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise InvalidStructuredResponseError(
                "The API returned an unexpected response shape; no table data could be parsed.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        if not text or not str(text).strip():
            raise InvalidStructuredResponseError(
                "The model returned an empty response; no table data could be parsed."
            )
        return str(text)


__all__ = ["OpenAICompatibleEngine"]

