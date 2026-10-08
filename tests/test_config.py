"""Tests for environment-driven configuration."""

from pdf2xlsx.config import Settings


def test_defaults_when_env_empty():
    s = Settings.from_env(env={})
    assert s.openai_api_key is None
    assert s.openai_model == "gpt-4o-mini"
    assert s.openai_base_url is None
    assert s.openai_temperature == 0.0
    assert s.ocr_model == "easyocr"
    assert s.ocr_languages == ("fa", "en")
    assert s.device == "auto"
    assert s.debug is False
    assert s.ocr_dpi == 200
    assert s.log_level == "INFO"


def test_openai_env_overrides():
    env = {
        "OPENAI_API_KEY": "sk-explicit",
        "OPENAI_MODEL": "gpt-4o",
        "OPENAI_BASE_URL": "https://gw.example/v1",
        "OPENAI_TEMPERATURE": "0.3",
    }
    s = Settings.from_env(env=env)
    assert s.openai_api_key == "sk-explicit"
    assert s.openai_model == "gpt-4o"
    assert s.openai_base_url == "https://gw.example/v1"
    assert s.openai_temperature == 0.3


def test_generic_wire_style_names_as_fallbacks():
    """API_KEY / BASE_URL / MODEL (gateway style) configure the OpenAI engine."""
    env = {
        "API_KEY": "sk-gateway-123",
        "BASE_URL": "http://localhost:20128/v1",
        "MODEL": "gemini/gemini-3.5-flash-lite",
    }
    s = Settings.from_env(env=env)
    assert s.openai_api_key == "sk-gateway-123"
    assert s.openai_base_url == "http://localhost:20128/v1"
    assert s.openai_model == "gemini/gemini-3.5-flash-lite"


def test_explicit_openai_names_win_over_generic():
    env = {
        "API_KEY": "sk-generic",
        "OPENAI_API_KEY": "sk-explicit",
        "MODEL": "generic-model",
        "OPENAI_MODEL": "explicit-model",
        "BASE_URL": "http://generic/v1",
        "OPENAI_BASE_URL": "http://explicit/v1",
    }
    s = Settings.from_env(env=env)
    assert s.openai_api_key == "sk-explicit"
    assert s.openai_model == "explicit-model"
    assert s.openai_base_url == "http://explicit/v1"


def test_env_overrides():
    env = {
        "OCR_MODEL": "Tesseract",
        "OCR_LANGUAGES": "fa , en , ar",
        "DEVICE": "CPU",
        "DEBUG": "true",
        "OCR_DPI": "300",
        "LOG_LEVEL": "debug",
    }
    s = Settings.from_env(env=env)
    assert s.ocr_model == "tesseract"
    assert s.ocr_languages == ("fa", "en", "ar")
    assert s.device == "cpu"
    assert s.debug is True
    assert s.ocr_dpi == 300
    assert s.log_level == "DEBUG"


def test_invalid_values_fall_back_to_defaults():
    s = Settings.from_env(env={"DEBUG": "maybe", "OCR_DPI": "abc", "OPENAI_TEMPERATURE": "x"})
    assert s.debug is False
    assert s.ocr_dpi == 200
    assert s.openai_temperature == 0.0


def test_api_key_never_in_repr():
    s = Settings.from_env(env={"OPENAI_API_KEY": "sk-secret-456"})
    assert "sk-secret-456" not in repr(s)
    assert "sk-secret-456" not in str(s)
