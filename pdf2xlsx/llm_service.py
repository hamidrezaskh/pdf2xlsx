"""LLM service: Gemini client + prompt + structured JSON + retry/validation.

Responsibilities:

* Gemini client construction (``google-generativeai`` SDK, vision-capable)
* prompt construction (reuses the generic extraction prompt in
  :mod:`pdf2xlsx.llm` — no domain-specific assumptions)
* JSON schema / structured output parsing into
  :class:`~pdf2xlsx.schemas.ExtractionPayload`
* validation of the parsed payload
* retry + exponential backoff + graceful error mapping (no secrets leak)

The OpenAI-compatible engine (``engines/openai_compat.py``) stays as the
generic gateway path; this module is the dedicated Gemini path so the
future TrOCR + Gemma-2B hybrid can reuse the same retry/parse helpers
(Gemma is also a Google model family with the same API shape).
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from .config import Settings
from .debug import DebugWriter
from .exceptions import ConfigurationError, InvalidStructuredResponseError, OpenAiApiError
from .llm import build_extraction_prompt, payload_from_text
from .schemas import ExtractionPayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# config / prompt helpers (pure, easily testable)
# ---------------------------------------------------------------------------


def resolve_gemini_settings(settings: Settings) -> dict:
    """Effective Gemini connection settings (explicit GEMINI_* or fallbacks)."""
    return {
        "api_key": settings.gemini_api_key or settings.openai_api_key,
        "model": settings.gemini_model or settings.openai_model,
        "temperature": float(settings.gemini_temperature),
        "max_retries": max(1, int(settings.gemini_max_retries)),
    }


def build_page_prompt(total_pages: int, page_number: int) -> str:
    return build_extraction_prompt(total_pages, page_number=page_number)


def build_document_prompt(total_pages: int) -> str:
    return build_extraction_prompt(total_pages, page_number=None)


def parse_structured_output(text: str) -> ExtractionPayload:
    """Parse + validate raw model text into the extraction payload."""
    return payload_from_text(text)


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------


@dataclass
class GeminiRequest:
    prompt: str
    png_bytes: Optional[bytes] = None
    mime_type: str = "image/png"


class GeminiClient:
    """Thin wrapper around the ``google-generativeai`` SDK with retries.

    ``client`` may be injected (tests) — otherwise it is built lazily from
    settings. All errors are mapped to :class:`OpenAiApiError`-compatible
    messages (reused type to avoid breaking existing error handling).
    """

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._cfg = resolve_gemini_settings(settings)

    @property
    def model_name(self) -> str:
        return str(self._cfg["model"])

    def _ensure_sdk_model(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._cfg["api_key"]
        if not api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your environment "
                "(or set OPENAI_BASE_URL to route through a gateway)."
            )
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ConfigurationError(
                "The 'google-generativeai' package is not installed. "
                "Run: pip install google-generativeai"
            ) from exc
        try:
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self._cfg["model"])
        except Exception as exc:
            raise OpenAiApiError(
                "Could not initialise the Gemini client.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        return self._client

    # -- low-level generate with retry ------------------------------------
    def generate_text(self, request: GeminiRequest) -> str:
        model = self._ensure_sdk_model()
        max_retries: int = self._cfg["max_retries"]
        temperature: float = self._cfg["temperature"]
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                parts: List[Any] = [request.prompt]
                if request.png_bytes:
                    # New SDK accepts blob dicts; older accepts inline_data.
                    parts.append(
                        {"mime_type": request.mime_type, "data": request.png_bytes}
                    )
                # Pass generation_config positionally-safe across SDK versions.
                try:
                    response = model.generate_content(
                        parts, generation_config={"temperature": temperature}
                    )
                except TypeError:
                    response = model.generate_content(parts)
                text = getattr(response, "text", "") or ""
                if not str(text).strip():
                    raise InvalidStructuredResponseError(
                        "The model returned an empty response; no table data could be parsed."
                    )
                return str(text)
            except InvalidStructuredResponseError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "gemini attempt %d/%d failed: %s: %s",
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise OpenAiApiError(
            "The Gemini API request failed after retries. Please try again.",
            detail=f"{type(last_exc).__name__}: {last_exc}" if last_exc else None,
        )

    # -- structured extraction ---------------------------------------------
    def extract_page_tables(
        self,
        png_bytes: bytes,
        *,
        page_number: int,
        total_pages: int,
        debug: Optional[DebugWriter] = None,
    ) -> ExtractionPayload:
        prompt = build_page_prompt(total_pages, page_number)
        text = self.generate_text(GeminiRequest(prompt=prompt, png_bytes=png_bytes))
        if debug is not None:
            debug.save_text(f"gemini_response_page_{page_number:03d}.txt", text)
        payload = parse_structured_output(text)
        # Trust our own pagination over the model's.
        for page in payload.pages:
            page.page_number = page_number
            for table in page.tables:
                table.page_number = page_number
        return payload


# ---------------------------------------------------------------------------
# OpenAI-compatible path helpers (shared retry wrapper for gateways,
# incl. Gemini-via-OpenAI-endpoint and the future Gemma-2B local server)
# ---------------------------------------------------------------------------


def base64_data_url(png_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


__all__ = [
    "GeminiRequest",
    "GeminiClient",
    "resolve_gemini_settings",
    "build_page_prompt",
    "build_document_prompt",
    "parse_structured_output",
    "base64_data_url",
]
