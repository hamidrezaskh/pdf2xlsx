"""Debug mode support.

When ``DEBUG`` is enabled, intermediate artifacts are written to a
timestamped run directory so extraction accuracy can be evaluated:

* page images
* raw OCR output
* raw model responses
* detected table structures / intermediate structured data
* the final XLSX
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DebugWriter:
    """Writes debug artifacts below ``<debug_dir>/<timestamp>/``."""

    def __init__(self, base_dir: str | Path = "debug_output", run_id: str | None = None) -> None:
        stamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(base_dir) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dir(self) -> Path:
        return self.run_dir

    def _write(self, name: str, data: bytes) -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug("debug artifact written: %s", path)
        return path

    def save_page_image(self, page_number: int, png_bytes: bytes) -> Path:
        return self._write(f"page_{page_number:03d}.png", png_bytes)

    def save_text(self, name: str, text: str) -> Path:
        return self._write(name, text.encode("utf-8"))

    def save_json(self, name: str, payload: Any) -> Path:
        try:
            if hasattr(payload, "model_dump"):
                data = payload.model_dump(mode="json")
            else:
                data = payload
            text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            text = f"<unserializable payload: {exc}>"
        return self._write(name, text.encode("utf-8"))

    def save_xlsx(self, name: str, data: bytes) -> Path:
        return self._write(name, data)
