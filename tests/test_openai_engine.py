"""Tests for the OpenAI-compatible engine (offline, with a fake client)."""

import json
from types import SimpleNamespace

import pytest

from pdf2xlsx.engines import OPENAI_MODE, engine_choices, get_engine
from pdf2xlsx.engines.openai_compat import (
    OpenAICompatibleEngine,
    _is_response_format_unsupported,
    _translate_api_error,
)
from pdf2xlsx.exceptions import (
    ConfigurationError,
    InvalidStructuredResponseError,
    OpenAiApiError,
)

# Shared structured payload (Persian/RTL table) reused by e2e tests.
SAMPLE_PAYLOAD = {
    "pages": [
        {
            "page_number": 1,
            "tables": [
                {
                    "table_index": 0,
                    "text_direction": "rtl",
                    "confidence": 0.95,
                    "columns": [
                        {"index": 0, "header": "ردیف"},
                        {"index": 1, "header": "شرح"},
                    ],
                    "rows": [
                        {
                            "index": 0,
                            "is_header": True,
                            "cells": [
                                {"row_index": 0, "column_index": 0, "value": "ردیف"},
                                {"row_index": 0, "column_index": 1, "value": "شرح"},
                            ],
                        },
                        {
                            "index": 1,
                            "cells": [
                                {"row_index": 1, "column_index": 0, "value": "۱"},
                                {"row_index": 1, "column_index": 1, "value": "پرداخت"},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
}

JSON_TEXT = json.dumps(SAMPLE_PAYLOAD)


# ---------------------------------------------------------------------------
# fake OpenAI-style client
# ---------------------------------------------------------------------------


class FakeCompletions:
    """behaviors: FIFO list of str (response text) or Exception to raise."""

    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.behaviors:
            raise AssertionError("no scripted behavior left")
        # FIFO, but the last behavior repeats (handy for multi-page tests)
        behavior = self.behaviors.pop(0) if len(self.behaviors) > 1 else self.behaviors[0]
        if isinstance(behavior, Exception):
            raise behavior
        message = SimpleNamespace(content=behavior)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(behaviors):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(behaviors))
    )


class FakePdf:
    """Minimal PdfDocument stand-in with page rendering."""

    def __init__(self, pages=1, name="fake.pdf"):
        self.page_count = pages
        self.source_name = name
        self.rendered = []

    def render_png(self, page_number, dpi=200):
        self.rendered.append((page_number, dpi))
        return b"\x89PNG\r\n\x1a\n" + b"fake-image"

    def read_bytes(self):  # pragma: no cover - unused here
        return b"%PDF-1.4"


def make_engine(make_settings, behaviors, **overrides):
    settings = make_settings(**overrides)
    return OpenAICompatibleEngine(settings, client=make_client(behaviors))


def http_error(status: int, message: str) -> Exception:
    err = RuntimeError(message)
    err.status_code = status  # type: ignore[attr-defined]
    return err


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def test_extract_single_page(make_settings):
    engine = make_engine(make_settings, [JSON_TEXT])
    result = engine.extract(FakePdf(pages=1))
    assert result.processing_mode == OPENAI_MODE
    assert result.model_name == engine.settings.openai_model
    assert result.table_count == 1
    page, table = next(result.iter_tables())
    assert page.page_number == 1
    assert table.text_direction == "rtl"
    assert [c.header for c in table.columns] == ["ردیف", "شرح"]
    assert result.warnings == []


def test_extract_multi_page_request_per_page(make_settings):
    engine = make_engine(make_settings, [JSON_TEXT, JSON_TEXT, JSON_TEXT])
    pdf = FakePdf(pages=3)
    result = engine.extract(pdf)
    assert len(result.pages) == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert pdf.rendered == [(1, 200), (2, 200), (3, 200)]
    calls = engine._client.chat.completions.calls
    assert len(calls) == 3  # one request per page


def test_page_numbering_is_forced_by_engine(make_settings):
    payload = {"pages": [{"page_number": 99, "tables": [{"table_index": 0}]}]}
    engine = make_engine(make_settings, [json.dumps(payload)])
    result = engine.extract(FakePdf(pages=2))
    # page 2's request may claim page 99 — engine rewrites it to 2
    assert result.pages[1].page_number == 2
    assert result.pages[1].tables[0].page_number == 2


def test_request_shape_uses_vision_and_json_mode(make_settings):
    engine = make_engine(make_settings, [JSON_TEXT])
    engine.extract(FakePdf(pages=1))
    call = engine._client.chat.completions.calls[0]
    assert call["model"] == engine.settings.openai_model
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "page 1 of 1" in content[0]["text"]  # per-page prompt
    assert "__SUBJECT__" not in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_fenced_json_response_parsed(make_settings):
    fenced = "```json\n" + JSON_TEXT + "\n```"
    engine = make_engine(make_settings, [fenced])
    assert engine.extract(FakePdf(pages=1)).table_count == 1


def test_response_format_fallback_for_plain_servers(make_settings):
    """Servers that reject JSON mode are retried without response_format."""
    unsupported = http_error(400, "response_format is not supported by this model")
    engine = make_engine(make_settings, [unsupported, JSON_TEXT])
    result = engine.extract(FakePdf(pages=1))
    assert result.table_count == 1
    calls = engine._client.chat.completions.calls
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_response_format_helper():
    assert _is_response_format_unsupported(http_error(400, "response_format json_object"))
    assert not _is_response_format_unsupported(http_error(400, "bad request"))
    assert not _is_response_format_unsupported(http_error(429, "response_format?"))


def test_per_page_failure_becomes_warning(make_settings):
    """One failed page must not abort the whole document."""
    engine = make_engine(make_settings, [JSON_TEXT, ConnectionError("connection refused")])
    result = engine.extract(FakePdf(pages=2))
    assert result.table_count == 1  # page 1 still extracted
    assert len(result.warnings) == 1
    assert "Page 2" in result.warnings[0]
    assert "Network" in result.warnings[0]
    assert result.pages[1].tables == []


def test_auth_error_message(make_settings):
    engine = make_engine(make_settings, [http_error(401, "Invalid API key")])
    result = engine.extract(FakePdf(pages=1))
    assert "OPENAI_API_KEY" in result.warnings[0]


def test_rate_limit_error_message(make_settings):
    engine = make_engine(make_settings, [http_error(429, "rate limit exceeded")])
    result = engine.extract(FakePdf(pages=1))
    assert "rate limit" in result.warnings[0].lower()


def test_empty_model_response_warning(make_settings):
    engine = make_engine(make_settings, ["   "])
    result = engine.extract(FakePdf(pages=1))
    assert any("empty" in w.lower() for w in result.warnings)


def test_invalid_json_response_warning(make_settings):
    engine = make_engine(make_settings, ["this is not json"])
    result = engine.extract(FakePdf(pages=1))
    assert any("not valid JSON" in w for w in result.warnings)


def test_missing_key_and_base_url_raises(make_settings):
    engine = OpenAICompatibleEngine(
        make_settings(openai_api_key=None, openai_base_url=None)
    )
    with pytest.raises(ConfigurationError) as exc:
        engine.extract(FakePdf(pages=1))
    assert "OPENAI_API_KEY" in str(exc.value)


def test_local_server_without_key_is_allowed(make_settings):
    """OPENAI_BASE_URL alone (e.g. Ollama) must not require a key."""
    settings = make_settings(
        openai_api_key=None, openai_base_url="http://localhost:11434/v1"
    )
    engine = OpenAICompatibleEngine(settings, client=make_client([JSON_TEXT]))
    assert engine.settings.openai_base_url.endswith("/v1")
    assert engine.extract(FakePdf(pages=1)).table_count == 1


def test_error_translator_shapes():
    net = _translate_api_error(ConnectionError("boom"))
    assert isinstance(net, OpenAiApiError) and "Network" in net.message
    auth = _translate_api_error(http_error(401, "nope"))
    assert "OPENAI_API_KEY" in auth.message
    quota = _translate_api_error(http_error(429, "slow down"))
    assert "rate limit" in quota.message.lower()
    generic = _translate_api_error(RuntimeError("weird"))
    assert "OpenAI-compatible API request failed" in generic.message


def test_debug_artifacts_saved(make_settings, tmp_path):
    from pdf2xlsx.debug import DebugWriter

    debug = DebugWriter(tmp_path / "dbg", run_id="openai_run")
    engine = make_engine(make_settings, [JSON_TEXT])
    engine.extract(FakePdf(pages=1), debug=debug)
    names = {p.name for p in debug.dir.iterdir()}
    assert "page_001.png" in names
    assert "openai_response_page_001.txt" in names


def test_engine_registered_in_registry(make_settings):
    choices = dict(engine_choices())
    assert choices[OPENAI_MODE] == "OpenAI-Compatible API"
    engine = get_engine("OpenAI Compatible API", make_settings())
    assert isinstance(engine, OpenAICompatibleEngine)

