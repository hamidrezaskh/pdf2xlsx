"""Application error hierarchy with user-friendly messages.

Messages are intentionally free of secrets (API keys, tokens, ...).
"""

from __future__ import annotations


class Pdf2XlsxError(Exception):
    """Base class for expected application errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidPdfError(Pdf2XlsxError):
    """The provided file is not a readable PDF."""


class EmptyPdfError(Pdf2XlsxError):
    """The PDF contains no pages."""


class PdfPasswordError(Pdf2XlsxError):
    """The PDF is encrypted and requires a password."""


class ConfigurationError(Pdf2XlsxError):
    """A required configuration value is missing or invalid."""


class OpenAiApiError(Pdf2XlsxError):
    """OpenAI-compatible API request failed (auth, network, quota, ...)."""


class OcrError(Pdf2XlsxError):
    """OCR engine could not be initialised or failed while running."""


class ExtractionError(Pdf2XlsxError):
    """The extraction engine failed to produce a result."""


class InvalidStructuredResponseError(Pdf2XlsxError):
    """The model returned a response that does not match the schema."""


class ExtractionValidationError(Pdf2XlsxError):
    """Structured data failed validation and cannot be converted."""


class XlsxGenerationError(Pdf2XlsxError):
    """XLSX generation failed."""


def friendly_message(exc: Exception) -> str:
    """Return a message that is safe to show in the UI."""
    if isinstance(exc, Pdf2XlsxError):
        return exc.message
    return (
        "An unexpected error occurred while processing the document. "
        "Please check the application logs for details."
    )
