"""Environment-driven configuration.

All configurable values come from environment variables (optionally loaded
from a ``.env`` file). Credentials are never hard-coded and are excluded
from reprs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # pragma: no cover - dotenv missing
    _load_dotenv = None

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return default


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip()) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(str(value).strip()) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    # OpenAI-compatible endpoints (OpenAI, OpenRouter, Ollama, vLLM, ...)
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_temperature: float = 0.0
    ocr_model: str = "easyocr"
    ocr_languages: tuple[str, ...] = ("fa", "en")
    device: str = "auto"
    ocr_dpi: int = 200
    debug: bool = False
    debug_dir: Path = Path("debug_output")
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables (loading ``.env`` if present)."""
        if env is None:
            if _load_dotenv is not None:
                try:
                    _load_dotenv()
                except Exception:  # pragma: no cover - never fatal
                    pass
            env = os.environ

        languages_raw = (env.get("OCR_LANGUAGES") or "fa,en").strip()
        languages = tuple(
            part.strip() for part in languages_raw.split(",") if part.strip()
        ) or ("fa", "en")

        return cls(
            # OpenAI-compatible engine. Generic wire-style names
            # (API_KEY / BASE_URL / MODEL) are accepted as fallbacks so
            # gateway setups (new-api/one-api style) work without renaming.
            openai_api_key=(
                (env.get("OPENAI_API_KEY") or env.get("API_KEY") or "").strip()
                or None
            ),
            openai_model=(
                (env.get("OPENAI_MODEL") or env.get("MODEL") or "").strip()
                or cls.openai_model
            ),
            openai_base_url=(
                (env.get("OPENAI_BASE_URL") or env.get("BASE_URL") or "").strip()
                or None
            ),
            openai_temperature=_as_float(env.get("OPENAI_TEMPERATURE"), 0.0),
            ocr_model=(env.get("OCR_MODEL") or "").strip().lower()
            or cls.ocr_model,
            ocr_languages=languages,
            device=(env.get("DEVICE") or "").strip().lower() or cls.device,
            ocr_dpi=max(72, _as_int(env.get("OCR_DPI"), 200)),
            debug=_as_bool(env.get("DEBUG"), False),
            debug_dir=Path((env.get("DEBUG_DIR") or "").strip() or "debug_output"),
            log_level=(env.get("LOG_LEVEL") or "").strip().upper() or cls.log_level,
        )
