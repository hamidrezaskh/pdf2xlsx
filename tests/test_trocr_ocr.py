"""Tests for the Persian TrOCR OCR backend (offline: fake recogniser)."""

import importlib.util

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pdf2xlsx.config import Settings
from pdf2xlsx.engines.opensource.ocr import (
    SUPPORTED_OCR_MODELS,
    create_ocr_backend,
)
from pdf2xlsx.engines.opensource.trocr_ocr import (
    TrocrBackend,
    align_tokens_to_boxes,
    band_word_boxes,
    binarize,
    mask_long_lines,
    segment_bands,
)

HAS_TRANSFORMERS = (
    importlib.util.find_spec("transformers") is not None
    and importlib.util.find_spec("torch") is not None
)


def _text_page(width=400, height=300, ruled=False) -> Image.Image:
    """White page with 3 text rows; optional full-width rules between them."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    rows = ["Code Label Amount", "A-1 Alpha 15", "B-2 Beta 42"]
    y = 40
    for i, text in enumerate(rows):
        draw.text((30, y), text, fill="black")
        y += 60
        if ruled and i < len(rows) - 1:
            draw.line([(0, y - 25), (width, y - 25)], fill="black", width=2)
    if ruled:  # table borders: full-width horizontal + full-height vertical
        draw.line([(0, 20), (width, 20)], fill="black", width=2)
        draw.line([(0, height - 15), (width, height - 15)], fill="black", width=2)
        draw.line([(10, 20), (10, height - 15)], fill="black", width=2)
        draw.line([(width - 10, 20), (width - 10, height - 15)], fill="black", width=2)
    return image


class SequenceRecognizer:
    """Fake recogniser: returns scripted lines top-to-bottom."""

    def __init__(self, lines):
        self.lines = list(lines)
        self.calls = 0

    def recognize(self, image):
        text = self.lines[min(self.calls, len(self.lines) - 1)]
        self.calls += 1
        return text, 0.9


# ---------------------------------------------------------------------------
# segmentation (pure numpy)
# ---------------------------------------------------------------------------


def test_binarize_finds_dark_ink():
    image = np.full((50, 50), 255, dtype=np.uint8)
    image[10:20, 5:45] = 0
    ink = binarize(image)
    assert ink[10:20, 5:45].all()
    assert not ink[30:40, :].any()


def test_mask_long_lines_removes_borders_keeps_text():
    image = _text_page(ruled=True)
    ink = binarize(np.asarray(image.convert("L"), dtype=np.uint8))
    clean, row_strips, col_strips = mask_long_lines(ink)
    assert row_strips, "horizontal borders should be detected"
    assert col_strips, "vertical borders should be detected"
    assert clean.sum() > 300  # text rows survive (PIL's font is tiny)
    assert clean[20:22, :].sum() == 0  # border row masked out


def test_segment_bands_finds_text_rows():
    image = _text_page(ruled=True)
    ink = binarize(np.asarray(image.convert("L"), dtype=np.uint8))
    clean, _, _ = mask_long_lines(ink)
    bands = segment_bands(clean)
    assert len(bands) == 3  # one band per text row (rules masked out)
    assert all(b.height >= 5 for b in bands)


def test_band_word_boxes_split_on_spaces():
    image = _text_page(ruled=False)
    ink = binarize(np.asarray(image.convert("L"), dtype=np.uint8))
    clean, _, _ = mask_long_lines(ink)
    bands = segment_bands(clean)
    boxes = band_word_boxes(clean, bands[1])  # "A-1 Alpha 15" -> 3 words
    assert len(boxes) == 3
    assert boxes[0][0] < boxes[1][0] < boxes[2][0]


# ---------------------------------------------------------------------------
# token <-> box alignment
# ---------------------------------------------------------------------------


def test_align_equal_counts_ltr():
    pairs = align_tokens_to_boxes("hello world", [(0, 10), (20, 30)])
    assert pairs == [("hello", (0, 10)), ("world", (20, 30))]


def test_align_rtl_reverses_visual_boxes():
    # logical "سلام دنیا": first token is the RIGHTMOST box visually
    pairs = align_tokens_to_boxes("سلام دنیا", [(0, 10), (20, 30)])
    assert pairs == [("سلام", (20, 30)), ("دنیا", (0, 10))]


def test_align_merges_extra_boxes():
    pairs = align_tokens_to_boxes("one two", [(0, 8), (9, 15), (30, 45)])
    assert len(pairs) == 2
    assert pairs[0][0] == "one" and pairs[1][0] == "two"


def test_align_splits_for_extra_tokens():
    pairs = align_tokens_to_boxes("a b c", [(0, 90)])
    assert len(pairs) == 3
    starts = [box[0] for _, box in pairs]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# backend with injected recogniser
# ---------------------------------------------------------------------------


def test_backend_produces_word_boxes():
    rec = SequenceRecognizer(["Code Label Amount", "A-1 Alpha 15", "B-2 Beta 42"])
    backend = TrocrBackend(recognizer=rec)
    words = backend.run(_text_page(ruled=True), page_number=1)
    texts = [w.text for w in words]
    assert texts[:3] == ["Code", "Label", "Amount"]
    assert len(words) == 9
    assert all(w.confidence == 0.9 for w in words)
    assert all(w.y1 > w.y0 and w.x1 > w.x0 for w in words)
    assert rec.calls == 3


def test_backend_empty_page_returns_no_words():
    empty = Image.new("RGB", (200, 200), "white")
    backend = TrocrBackend(recognizer=SequenceRecognizer(["ignored"]))
    assert backend.run(empty, page_number=1) == []


def test_backend_feeds_engine_pipeline(make_settings, tmp_path):
    """TrocrBackend output works as the engine's OCR source (fake model)."""
    from conftest import BORDERLESS_TABLE, build_scanned_pdf, words_from_borderless_spec
    from pdf2xlsx.engines.opensource import OpenSourceEngine

    scanned = build_scanned_pdf(tmp_path / "scan.pdf", [[BORDERLESS_TABLE]])
    scale = 150.0 / 72.0
    spec_lines = [ " ".join(row) for row in BORDERLESS_TABLE.rows]
    rec = SequenceRecognizer(spec_lines)
    backend = TrocrBackend(recognizer=rec)
    engine = OpenSourceEngine(make_settings(ocr_model="trocr"), ocr_backend=backend)
    with __import__("pdf2xlsx.pdf", fromlist=["load_pdf"]).load_pdf(scanned) as pdf:
        result = engine.extract(pdf)
    assert result.table_count == 1
    _, table = next(result.iter_tables())
    assert table.n_columns == 3


# ---------------------------------------------------------------------------
# factory / config / dependency guards
# ---------------------------------------------------------------------------


def test_factory_creates_trocr_backend():
    backend = create_ocr_backend(Settings(ocr_model="trocr", device="cpu"))
    assert isinstance(backend, TrocrBackend)
    assert backend.model_id == "hezarai/trocr-base-fa-v2"
    assert "trocr" in SUPPORTED_OCR_MODELS


def test_trocr_model_id_configurable():
    backend = create_ocr_backend(
        Settings(ocr_model="trocr", device="cpu", trocr_model="custom/other")
    )
    assert backend.model_id == "custom/other"


def test_trocr_env_override():
    s = Settings.from_env(env={"OCR_MODEL": "trocr", "TROCR_MODEL": "my/model"})
    assert s.ocr_model == "trocr"
    assert s.trocr_model == "my/model"


@pytest.mark.skipif(HAS_TRANSFORMERS, reason="dependencies installed; loader would download")
def test_missing_dependencies_raise_friendly_error():
    from pdf2xlsx.exceptions import OcrError

    backend = TrocrBackend()  # no injected recogniser
    with pytest.raises(OcrError) as exc:
        backend.run(_text_page(), page_number=1)
    assert "requirements-ocr.txt" in str(exc.value)

