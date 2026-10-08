"""Tests for the engine registry and dynamic UI mode discovery."""

import pytest

from pdf2xlsx.engines import (
    OPENAI_MODE,
    OPENSOURCE_MODE,
    ExtractionEngine,
    available_engines,
    engine_choices,
    get_engine,
    normalize_mode,
    register_engine,
)
from pdf2xlsx.exceptions import ConfigurationError
from pdf2xlsx.schemas import ExtractionResult


def test_builtin_engines_registered():
    names = available_engines()
    assert OPENAI_MODE in names
    assert OPENSOURCE_MODE in names


def test_engine_choices_provides_ui_labels():
    choices = dict(engine_choices())
    assert choices[OPENAI_MODE] == "OpenAI-Compatible API"
    assert choices[OPENSOURCE_MODE] == "Open-Source Model"
    # every choice has a non-empty label
    assert all(label for label in choices.values())


def test_engine_choices_matches_available_engines():
    assert sorted(dict(engine_choices())) == available_engines()


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("openai", "openai"),
        ("OpenAI-Compatible API", "openai"),
        ("openai compatible", "openai"),
        ("OpenAI API", "openai"),
        ("Open-Source Model", "opensource"),
        ("open source", "opensource"),
        ("ocr", "opensource"),
        ("opensource", "opensource"),
    ],
)
def test_normalize_mode_aliases(alias, expected):
    assert normalize_mode(alias) == expected


def test_get_engine_builds_instances(make_settings):
    engine = get_engine(OPENSOURCE_MODE, make_settings())
    assert isinstance(engine, ExtractionEngine)
    assert engine.name == OPENSOURCE_MODE


def test_unknown_mode_lists_available(make_settings):
    with pytest.raises(ConfigurationError) as exc:
        get_engine("quantum-processor", make_settings())
    message = str(exc.value)
    assert "quantum-processor" in message
    assert OPENAI_MODE in message
    assert OPENSOURCE_MODE in message


def test_registering_engine_appears_in_ui_choices(make_settings):
    """A new processor must show up without touching the Streamlit UI."""
    from pdf2xlsx.engines import _ENGINES

    @register_engine
    class FutureProcessor(ExtractionEngine):
        name = "future-test"
        description = "Future Processor"

        def extract(self, pdf, *, debug=None):
            return ExtractionResult()  # pragma: no cover

    try:
        assert "future-test" in available_engines()
        assert dict(engine_choices())["future-test"] == "Future Processor"
        engine = get_engine("future-test", make_settings())
        assert isinstance(engine, FutureProcessor)
    finally:
        _ENGINES.pop("future-test", None)  # do not leak into other tests
