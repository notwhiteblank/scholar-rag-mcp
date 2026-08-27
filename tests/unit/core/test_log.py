import logging

import pytest

from scholar_rag.core import log as log_module


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SCHOLAR_RAG_LOG_DIR", raising=False)
    monkeypatch.delenv("SCHOLAR_RAG_LOG_LEVEL", raising=False)
    log_module._settings = None
    log_module._file_log_configured = False


@pytest.fixture(autouse=True)
def _reset_file_handler():
    yield
    root = logging.getLogger()
    root.handlers[:] = [h for h in root.handlers if not isinstance(h, logging.FileHandler)]
    root.setLevel(logging.WARNING)
    log_module._settings = None
    log_module._file_log_configured = False


def test_no_log_dir_writes_no_file(tmp_path):
    logger = log_module.get_logger("test.nofile")
    logger.warning("no file expected")
    assert list(tmp_path.iterdir()) == []


def test_log_dir_writes_debug_file(monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("SCHOLAR_RAG_LOG_DIR", str(log_dir))
    monkeypatch.setenv("SCHOLAR_RAG_LOG_LEVEL", "WARNING")
    logger = log_module.get_logger("test.filelog")
    logger.debug("debug probe")
    logger.warning("warning probe")
    target = log_dir / "scholar-rag.log"
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "debug probe" in content
    assert "warning probe" in content


def test_log_dir_created_lazily_only_on_logger_use(monkeypatch, tmp_path):
    log_dir = tmp_path / "not_yet_created"
    monkeypatch.setenv("SCHOLAR_RAG_LOG_DIR", str(log_dir))
    assert not log_dir.exists()
    log_module.get_logger("test.lazy")
    assert log_dir.is_dir()
