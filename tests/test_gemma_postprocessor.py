"""Tests for the TrOCR + Gemma 2B hybrid (offline: fake models)."""

import json

import pytest

from pdf2xlsx.config import Settings
from pdf2xlsx.engines.opensource.gemma_postprocessor import (
    GemmaCorrector,
    NoOpCorrector,
    build_correction_prompt,
    create_corrector,
    parse_corrected_lines,
)
from pdf2xlsx.engines.opensource.ocr import create_ocr_backend
from pdf2xlsx.engines.opensource.trocr_ocr import TrocrBackend
from pdf2xlsx.exceptions import ConfigurationError


class FakeCorrector:
    """Deterministic stand-in: appends a marker to each line."""

    def __init__(self, suffix=" [fixed]"):
        self.suffix = suffix
        self.calls = []

    def correct_lines(self, lines):
        self.calls.append(list(lines))
        return [t + self.suffix for t in lines]


class FailingCorrector:
    def correct_lines(self, lines):
        raise RuntimeError("boom")


class SequenceRecognizer:
    def __init__(self, lines):
        self.lines = list(lines)

    def recognize(self, image):
        from PIL import Image as _Image  # noqa

        text = self.lines.pop(0) if self.lines else ""
        return text, 0.9


def _image_with_lines(n=2):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (400, 60 * n + 40), "white")
    draw = ImageDraw.Draw(image)
    for i in range(n):
        draw.text((30, 30 + i * 60), f"row {i}", fill="black")
    return image


# -- prompt / parse (pure) --------------------------------------------------


def test_prompt_contains_all_lines_as_json():
    lines = ["سلام دنیا", "مبلغ ۱۲۳"]
    prompt = build_correction_prompt(lines)
    assert "سلام دنیا" in prompt
    decoded = json.loads(prompt[prompt.index("[") : prompt.rindex("]") + 1])
    assert decoded == lines


def test_parse_valid_json_array():
    raw = 'noise ["a ب", "c ۱۲۳"] trailing'
    assert parse_corrected_lines(raw, 2, ["x", "y"]) == ["a ب", "c ۱۲۳"]


def test_parse_mismatch_falls_back_to_original():
    assert parse_corrected_lines('["only-one"]', 2, ["a", "b"]) == ["a", "b"]
    assert parse_corrected_lines("", 2, ["a", "b"]) == ["a", "b"]


# -- factory ----------------------------------------------------------------


def test_factory_none_is_noop():
    assert isinstance(create_corrector("none"), NoOpCorrector)
    assert create_corrector("none").correct_lines(["a"]) == ["a"]


def test_factory_gemma_aliases():
    for key in ("gemma", "trocr+gemma", "trocr_gemma"):
        c = create_corrector(key, model_id="m", device="cpu")
        assert isinstance(c, GemmaCorrector)
        assert c.model_id == "m"


def test_factory_unknown_raises():
    with pytest.raises(ConfigurationError):
        create_corrector("bert")


def test_gemma_corrector_batches_with_injected_model():
    """Batching + parsing work without downloading weights."""

    class StubTokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors=None):
            import torch

            return {"input_ids": torch.zeros((1, 3), dtype=torch.long)}

        def decode(self, tokens, skip_special_tokens=True):
            return self._next

    class StubModel:
        device = "cpu"

        def __init__(self, tok):
            self._tok = tok

        def generate(self, **kwargs):
            import torch

            base = kwargs["input_ids"]
            extra = torch.zeros((1, 2), dtype=torch.long)
            return torch.cat([base, extra], dim=1)

    tok = StubTokenizer()
    batches = [['["l1 fixed", "l2 fixed"]'], ['["l3 fixed"]']]
    c = GemmaCorrector(model_id="m", batch_size=2, model=StubModel(tok), tokenizer=tok)
    texts, calls = [], 0

    def fake_generate(prompt):
        nonlocal calls
        tok._next = batches[calls][0]
        calls += 1
        return tok._next

    c._generate = fake_generate  # type: ignore[method-assign]
    assert c.correct_lines(["l1", "l2", "l3"]) == ["l1 fixed", "l2 fixed", "l3 fixed"]
    assert calls == 2


# -- TrocrBackend integration ------------------------------------------------


def test_trocr_backend_applies_corrector_before_alignment():
    rec = SequenceRecognizer(["hello world", "foo bar"])
    backend = TrocrBackend(recognizer=rec, corrector=FakeCorrector(suffix="!"))
    words = backend.run(_image_with_lines(2), page_number=1)
    assert words, "expected words from two bands"
    # corrector rewrites whole lines ("hello world!" -> tokens ["hello", "world!"])
    assert len(words) == 4
    assert [w.text for w in words].count("hello") == 1
    assert any(w.text.endswith("!") for w in words)
    assert all(w.confidence == 0.9 for w in words)  # geometry/conf preserved


def test_trocr_backend_failing_corrector_keeps_trocr_text():
    rec = SequenceRecognizer(["hello world", "foo bar"])
    backend = TrocrBackend(recognizer=rec, corrector=FailingCorrector())
    words = backend.run(_image_with_lines(2), page_number=1)
    assert {w.text for w in words} >= {"hello", "world", "foo", "bar"}


# -- backend factory wiring --------------------------------------------------


def test_factory_trocr_plain_has_no_corrector():
    backend = create_ocr_backend(Settings(ocr_model="trocr", device="cpu"))
    assert isinstance(backend, TrocrBackend)
    assert backend._corrector is None


def test_factory_trocr_gemma_shorthand_attaches_corrector():
    backend = create_ocr_backend(Settings(ocr_model="trocr+gemma", device="cpu"))
    assert isinstance(backend, TrocrBackend)
    assert isinstance(backend._corrector, GemmaCorrector)


def test_factory_trocr_with_postprocessor_env():
    s = Settings.from_env(
        env={"OCR_MODEL": "trocr", "OCR_POSTPROCESSOR": "gemma", "GEMMA_MODEL": "m/gemma"}
    )
    backend = create_ocr_backend(s)
    assert isinstance(backend, TrocrBackend)
    assert backend._corrector.model_id == "m/gemma"
