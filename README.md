# PDF → XLSX

A generic PDF-to-XLSX converter with **OCR / document understanding** and
**fully dynamic table extraction**. Tables, rows, columns, column names,
cell boundaries and layouts are all inferred from the input document —
nothing about the table structure is hard-coded.

```
PDF → page/image processing → extraction engine → dynamic table detection
    → normalized structured data (Pydantic) → validation → XLSX (openpyxl)
```

## Features

- **Two processing modes** behind one interface:
  1. **OpenAI-Compatible API** — any OpenAI Chat Completions endpoint:
     OpenAI, OpenRouter, new-api/one-api gateways, Ollama, vLLM,
     LM Studio, ... Each page is sent as an image with JSON mode
     (transparently retried without `response_format` for servers that
     don't support it). Gemini models are reached through such gateways.
  2. **Open-Source OCR / Document Understanding** — local pipeline:
     embedded PDF text when available, OCR (EasyOCR/Tesseract) for
     scanned pages, then dynamic table structure detection.
- **Dynamic table detection** — ruling-line grid detection for bordered
  tables, plus whitespace/alignment detection for borderless tables.
- **Multilingual: Persian + English** — RTL/LTR text, mixed-language
  content, Persian/Arabic/Latin digits, Unicode. Text is never
  destructively normalized.
- **Normalized intermediate model** shared by both engines
  (`pdf2xlsx/schemas.py`): pages → tables → columns/rows → cells with
  bounding boxes, confidence, merged cells, text direction.
- **XLSX generation** independent from the engines: one sheet per table,
  dynamic columns/rows, empty cells, merged cells, RTL sheets, header
  styling, auto-sized columns, Unicode-safe.
- **Streamlit UI**: upload → mode selection → convert → live status →
  preview → download.
- **Validation layer** normalizes any engine output (contiguous indexes,
  padded ragged rows, de-duplicated cells, clamped spans, derived
  merged-cell regions, header propagation).
- **Debug mode**: saves page images, OCR output, raw model responses,
  intermediate structured data and the final XLSX.
- **Logging** of mode, pages, durations, statuses and errors
  (no secrets).

## Installation

```bash
# core (OpenAI-compatible + open-source mode on text-layer PDFs)
pip install -r requirements.txt

# optional: OCR extras for scanned/image-only PDFs (pulls in PyTorch)
pip install -r requirements-ocr.txt
```

Python ≥ 3.10.

## Configuration

Copy `.env.example` to `.env` and adjust. All values come from the
environment — credentials are never hard-coded.

| Variable             | Default            | Purpose                              |
| -------------------- | ------------------ | ------------------------------------ |
| `OPENAI_API_KEY`     | *(unset)*          | Key for the OpenAI-compatible API *(alias: `API_KEY`)* |
| `OPENAI_BASE_URL`    | *(unset)*          | Endpoint base URL, e.g. `http://localhost:20128/v1` *(alias: `BASE_URL`)* |
| `OPENAI_MODEL`       | `gpt-4o-mini`      | Model name, e.g. `gemini/gemini-3.5-flash-lite` behind a gateway *(alias: `MODEL`)* |
| `OPENAI_TEMPERATURE` | `0`                | Sampling temperature                 |
| `OCR_MODEL`          | `easyocr`          | `easyocr` \| `trocr` \| `tesseract` \| `native` |
| `TROCR_MODEL`        | `hezarai/trocr-base-fa-v2` | Persian TrOCR line model (Hugging Face id), used by `OCR_MODEL=trocr` |
| `OCR_LANGUAGES`      | `fa,en`            | OCR languages (Persian + English)    |
| `DEVICE`             | `auto`             | `auto` \| `cpu` \| `cuda`            |
| `OCR_DPI`            | `200`              | Render DPI before OCR                |
| `DEBUG`              | `false`            | Save intermediate artifacts          |
| `DEBUG_DIR`          | `debug_output`     | Where debug artifacts go             |
| `LOG_LEVEL`          | `INFO`             | Logging level                        |

## Usage

```bash
streamlit run app.py
```

1. Upload a PDF.
2. Choose **OpenAI-Compatible API** or **Open-Source Model**.
3. Click **Convert to XLSX** — progress, validation issues and a table
   preview appear.
4. Download the workbook (one sheet per table: `P{page}T{index}`).

### OpenAI-compatible endpoints

The **OpenAI-Compatible API** mode talks to any endpoint implementing
the Chat Completions API. Examples for `.env`:

```bash
# a new-api / one-api style gateway serving Gemini models
API_KEY=sk-...
BASE_URL=http://localhost:20128/v1
MODEL=gemini/gemini-3.5-flash-lite

# local Ollama (no key required — BASE_URL alone is enough)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen2.5vl

# official OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Notes:

- Pages are sent as **base64 PNG images** (resolution: `OCR_DPI`), so the
  model must be vision-capable.
- Each page is one request; a failing page becomes a warning instead of
  aborting the document.
- Generic names `API_KEY` / `BASE_URL` / `MODEL` are accepted as aliases;
  explicit `OPENAI_*` names win when both are set.

## Architecture

```
app.py                       Streamlit UI (UI concerns only)
└── pdf2xlsx/
    ├── config.py            env-driven Settings (no secrets in repr)
    ├── schemas.py           normalized Pydantic data model
    ├── validation.py        normalize/validate any engine's output
    ├── textutil.py          RTL/LTR analysis, logical text assembly
    ├── llm.py               shared LLM prompt + JSON payload parsing
    ├── pdf/processor.py     PDF loading, rawdict words, ruling lines, rendering
    ├── engines/
    │   ├── __init__.py      registry: get_engine(mode, settings)
    │   ├── base.py          ExtractionEngine interface
    │   ├── openai_compat.py OpenAI-compatible vision API (page images + JSON mode)
    │   └── opensource/      OCR backends + layout/structure detection
    │       ├── ocr.py       easyocr / tesseract / native (lazy, cached)
    │       ├── layout.py    line clustering, grids, whitespace columns
    │       └── structure.py words → dynamic rows/columns/cells
    ├── xlsx/generator.py    openpyxl; consumes ONLY the normalized model
    ├── pipeline.py          orchestration + logging + progress + debug
    ├── debug.py             DebugWriter artifacts
    ├── logging_config.py
    └── exceptions.py        user-friendly, secret-free errors
tests/                       offline test suite (fixture PDFs built on the fly)
```

### Pipeline

```
PDF
→ PDF processing (load/validate, page images, native text, ruling lines)
→ Extraction engine (OpenAI-compatible | Open-Source)   ← same interface
→ Table structure detection (ruling lines | whitespace)
→ Normalized structured data (ExtractionResult)
→ Validation / normalization (ValidationReport)
→ XLSX generation (openpyxl)
```

## Adding a new processor

Implement the interface and register it — the UI and XLSX layer stay
untouched:

```python
from pdf2xlsx.engines import register_engine
from pdf2xlsx.engines.base import ExtractionEngine

@register_engine
class FutureProcessor(ExtractionEngine):
    name = "future"
    description = "Future processor"

    def extract(self, pdf, *, debug=None):
        ...  # return ExtractionResult
```

`get_engine("future", settings)` and the UI's engine list pick it up
automatically.

## How the open-source mode works

1. **Text source per page** — the embedded text layer is used when
   present (exact, free). Otherwise the page is rendered and the
   configured OCR backend runs (loaded lazily, cached across requests):
   - `easyocr` (default) — Persian + English,
   - `trocr` — [hezarai/trocr-base-fa-v2](https://huggingface.co/hezarai/trocr-base-fa-v2),
     a Persian TrOCR *line* model; the backend adds its own page →
     lines → words segmentation (numpy-only: Otsu binarization, ruling
     line masking, projection bands) and aligns tokens to word boxes
     (RTL-aware). Weights (~935 MB) are downloaded from Hugging Face on
     first use; model id configurable via `TROCR_MODEL`.
   - `tesseract`, or `native` (OCR disabled).
2. **Table detection** — two complementary detectors:
   - *Ruling lines*: vector lines in the PDF are clustered and connected
     into grids (≥2×2 cells) via intersection components.
   - *Whitespace*: lines are grouped into blocks; vertical gaps that are
     consistent across ≥75 % of the lines become column separators
     (borderless tables). Prose almost never satisfies this.
3. **Structure reconstruction** — words are assigned to (row, column)
   bands by their centers; wrapped lines inside one column merge into a
   single row; rows/columns/headers are inferred (a header is only
   claimed when a text header sits above at least one numeric column —
   conservative, so no data is ever dropped).

### RTL / multilingual handling

- Words are rebuilt from per-character bounding boxes: glyph positions
  in a PDF are always visual, so sorting a word's characters by *x*
  yields visual order; for RTL words this is reversed back to logical
  order, while embedded digit/Latin runs are restored.
- Arabic **presentation forms** (a PDF shaping artifact) are mapped back
  to canonical letters with NFKC — applied *only* to those code point
  ranges; no other text is normalized.
- Line majority direction provides the bidi context for mixed words.
- OCR backends return their own logical reading order and are trusted
  as provided.

## Debug mode

With `DEBUG=true` (or an explicit `DebugWriter`), each run writes to
`debug_output/<timestamp>/`:

- `page_001.png` — rendered pages
- `ocr_page_001.json` — raw OCR word boxes + confidences
- `openai_response_page_001.txt` — raw model output
- `tables_page_001.json` — detected table structures
- `structured_result.json` — normalized result
- `result.xlsx` — final workbook

## Testing

```bash
pytest
```

The suite is fully offline: fixture PDFs (ruled tables, borderless
tables, multi-page/multi-layout documents, prose controls, scanned
image-only PDFs, encrypted/corrupt files, Persian tables) are generated
programmatically with PyMuPDF; the LLM engine and OCR are exercised
through injected fakes. Covered: PDF processing, layout detection,
table reconstruction, structured-data validation, LLM response parsing,
open-source processing, XLSX generation, multiple tables/pages, dynamic
structures, empty cells, merged cells and end-to-end runs.

## Error handling

All expected failures raise `Pdf2XlsxError` subclasses with
user-friendly, secret-free messages (invalid/corrupt/empty/encrypted
PDFs, missing API key, API/network/quota errors, OCR failures, invalid
model responses, XLSX errors). The Streamlit layer shows them via
`st.error`; details go to the log. A page-level OCR failure becomes a
warning instead of aborting the whole document.

## Known limitations

- Word-level reconstruction assumes PDFs produced by bidi-aware writers
  (browsers, Word, LaTeX, PyMuPDF's HTML engine). Files whose glyphs
  are written in non-bidi logical order may need post-processing.
- Header detection is deliberately conservative (requires numeric
  contrast below the header row) so first data rows are never mistaken
  for headers; in OpenAI-compatible mode headers are detected semantically.
- Single-column borderless tables are not detected (need ≥1 column
  separator).

## License

Use and modify freely for your projects.

