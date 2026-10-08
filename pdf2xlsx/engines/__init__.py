"""Extraction engine registry.

Engines register themselves under a mode name; the UI and pipeline only
deal with the registry, never with concrete implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

from ..exceptions import ConfigurationError
from .base import ExtractionEngine

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

OPENAI_MODE = "openai"
OPENSOURCE_MODE = "opensource"

_ENGINES: Dict[str, Type[ExtractionEngine]] = {}

_MODE_ALIASES = {
    "openai": OPENAI_MODE,
    "openai api": OPENAI_MODE,
    "openai api compatible": OPENAI_MODE,
    "openai-compatible": OPENAI_MODE,
    "openai compatible": OPENAI_MODE,
    "openai-compatible api": OPENAI_MODE,
    "openai compatible api": OPENAI_MODE,
    "chatgpt": OPENAI_MODE,
    "open-source": OPENSOURCE_MODE,
    "open source": OPENSOURCE_MODE,
    "open-source model": OPENSOURCE_MODE,
    "open source model": OPENSOURCE_MODE,
    "opensource model": OPENSOURCE_MODE,
    "ocr": OPENSOURCE_MODE,
}

_LOADED = False


def register_engine(cls: Type[ExtractionEngine]) -> Type[ExtractionEngine]:
    """Class decorator: add an engine to the registry."""
    if not cls.name:
        raise ValueError("engine class must define a non-empty 'name'")
    _ENGINES[cls.name] = cls
    return cls


def _ensure_loaded() -> None:
    """Import engine modules (side effect: registration)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from . import openai_compat as _openai_compat  # noqa: F401
    from . import opensource as _opensource  # noqa: F401


def available_engines() -> list[str]:
    _ensure_loaded()
    return sorted(_ENGINES)


def engine_choices() -> list[tuple[str, str]]:
    """Ordered ``[(engine name, display label)]`` pairs for UIs.

    Labels come from each engine's ``description`` so newly registered
    processors appear in the UI without touching it.
    """
    _ensure_loaded()
    return [
        (name, _ENGINES[name].description or name) for name in sorted(_ENGINES)
    ]


def normalize_mode(mode: str) -> str:
    key = (mode or "").strip().lower()
    return _MODE_ALIASES.get(key, key)


def get_engine(mode: str, settings: "Settings") -> ExtractionEngine:
    """Instantiate the engine for ``mode`` (raises ConfigurationError)."""
    _ensure_loaded()
    key = normalize_mode(mode)
    engine_cls = _ENGINES.get(key)
    if engine_cls is None:
        raise ConfigurationError(
            f"Unknown processing mode '{mode}'. Available modes: "
            f"{', '.join(sorted(_ENGINES))}."
        )
    return engine_cls(settings)


__all__ = [
    "ExtractionEngine",
    "OPENAI_MODE",
    "OPENSOURCE_MODE",
    "register_engine",
    "get_engine",
    "available_engines",
    "engine_choices",
    "normalize_mode",
]
