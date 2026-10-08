"""Tests for logging setup (idempotency, file output, encoding safety)."""

import logging


def test_setup_logging_is_idempotent():
    from pdf2xlsx.logging_config import setup_logging

    setup_logging("INFO")
    count_after_first = len(logging.getLogger().handlers)
    setup_logging("DEBUG")
    count_after_second = len(logging.getLogger().handlers)
    assert count_after_first == count_after_second  # no duplicate handlers
    # level still applied on repeat calls
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_writes_file(tmp_path, monkeypatch):
    import pdf2xlsx.logging_config as lc

    root = logging.getLogger()
    before = list(root.handlers)
    monkeypatch.setattr(lc, "_CONFIGURED", False)  # force re-configuration
    log_file = tmp_path / "app.log"
    try:
        lc.setup_logging("INFO", log_file)
        added = [h for h in root.handlers if h not in before]
        record_file = [
            h for h in added
            if isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(log_file)
        ]
        assert record_file, "file handler was not installed"
        lc.get_logger("pdf2xlsx.test").info("pipeline started %s", "ok")
        for handler in added:
            handler.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "pipeline started ok" in content
    finally:
        # remove only the handlers this test added
        for handler in [h for h in root.handlers if h not in before]:
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # pragma: no cover
                pass


def test_non_ascii_log_records_do_not_crash():
    """Persian text must survive on restrictive Windows consoles."""
    from pdf2xlsx.logging_config import get_logger, setup_logging

    setup_logging("INFO")
    logger = get_logger("pdf2xlsx.encoding-test")
    logger.info("اعتبارسنجی: %s — %d table(s)", "جدول", 3)  # must not raise
