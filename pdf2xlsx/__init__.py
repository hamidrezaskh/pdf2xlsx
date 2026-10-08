"""Generic PDF -> XLSX conversion package.

Pipeline: PDF -> page/image processing -> extraction engine
(OpenAI-compatible API or open-source OCR / document understanding)
-> dynamic table structure detection -> normalized structured data
-> validation -> XLSX generation.

Nothing about table structure (columns, rows, headers, dimensions) is
hard-coded; everything is inferred from the input document.
"""

__version__ = "1.0.0"
