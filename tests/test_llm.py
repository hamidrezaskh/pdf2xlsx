"""Tests for shared LLM helpers: extraction prompt + JSON payload parsing."""

import json

import pytest

from pdf2xlsx.exceptions import InvalidStructuredResponseError
from pdf2xlsx.llm import build_extraction_prompt, payload_from_text, strip_code_fences

from test_openai_engine import SAMPLE_PAYLOAD

SAMPLE_TEXT = json.dumps(SAMPLE_PAYLOAD)


def test_prompt_for_whole_pdf_is_generic():
    prompt = build_extraction_prompt(7)
    assert "7" in prompt
    assert "__SUBJECT__" not in prompt
    assert "__SCOPE__" not in prompt
    assert "column" in prompt.lower()
    # no hard-coded schema assumptions
    assert "NEVER assume" in prompt


def test_prompt_targets_single_page_image():
    prompt = build_extraction_prompt(10, page_number=3)
    assert "page 3 of 10" in prompt
    assert "Return ONLY tables visible on page 3" in prompt
    assert "__SUBJECT__" not in prompt
    assert "NEVER assume" in prompt


def test_strip_code_fences():
    assert strip_code_fences("```json\n{}```") == "{}"
    assert strip_code_fences("{\"a\": 1}") == "{\"a\": 1}"


def test_payload_from_plain_json():
    payload = payload_from_text(SAMPLE_TEXT)
    assert payload.pages[0].tables[0].rows[1].cells[1].value == "پرداخت"


def test_payload_from_fenced_json():
    fenced = "```json\n" + SAMPLE_TEXT + "\n```"
    payload = payload_from_text(fenced)
    assert payload.pages[0].page_number == 1


def test_payload_rejects_invalid_json():
    with pytest.raises(InvalidStructuredResponseError):
        payload_from_text("this is not json at all")


def test_payload_rejects_wrong_shape():
    with pytest.raises(InvalidStructuredResponseError):
        payload_from_text(json.dumps({"pages": [{"page_number": "abc"}]}))


def test_payload_rejects_empty():
    with pytest.raises(InvalidStructuredResponseError):
        payload_from_text("   ")
