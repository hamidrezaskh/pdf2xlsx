"""Application logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once; safe to call repeatedly."""
    global _CONFIGURED
    # Windows consoles (cp1252 & friends) must not crash on non-ASCII
    # (e.g. Persian) log records.
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(errors="replace")
    except Exception:  # pragma: no cover - stream may be a wrapper
        pass

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    if _CONFIGURED:
        return

    formatter = logging.Formatter(_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        try:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:  # pragma: no cover - logging must never break the app
            pass

    # Quiet noisy third-party loggers
    for name in ("PIL", "matplotlib", "urllib3", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
