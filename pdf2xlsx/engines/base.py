"""Common interface implemented by every extraction engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..debug import DebugWriter
    from ..pdf import PdfDocument
    from ..schemas import ExtractionResult


class ExtractionEngine(ABC):
    """Converts a PDF into the normalized structured representation.

    The OpenAI-compatible processor and the open-source OCR processor
    implement this interface and therefore produce the same output shape.
    Adding a new processor requires only a new subclass + registration;
    the Streamlit UI and the XLSX generator stay untouched.
    """

    #: registry key / processing mode identifier, e.g. "openai"
    name: ClassVar[str] = "engine"
    #: human-readable label for the UI
    description: ClassVar[str] = ""

    def __init__(self, settings: "Settings") -> None:
        self.settings = settings

    @property
    def model_name(self) -> Optional[str]:
        """Name of the underlying model, if any (for logging / metadata)."""
        return None

    @abstractmethod
    def extract(
        self, pdf: "PdfDocument", *, debug: Optional["DebugWriter"] = None
    ) -> "ExtractionResult":
        """Run extraction. Raises Pdf2XlsxError subclasses on failure."""

    def __repr__(self) -> str:  # never expose credentials
        return f"<{type(self).__name__} model={self.model_name!r}>"
