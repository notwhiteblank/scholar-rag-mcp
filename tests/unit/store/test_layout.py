import os
from pathlib import Path

import pytest

from scholar_rag.core.config import Settings
from scholar_rag.store import layout


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", qdrant_storage_dir=tmp_path / "qdrant")


def test_kbs_root(tmp_path):
    assert layout.kbs_root(_settings(tmp_path)) == tmp_path / "data" / "kbs"


def test_kb_dir(tmp_path):
    assert layout.kb_dir("alpha", _settings(tmp_path)) == tmp_path / "data" / "kbs" / "alpha"


def test_catalog_path(tmp_path):
    assert layout.catalog_path("alpha", _settings(tmp_path)) == tmp_path / "data" / "kbs" / "alpha" / "catalog.sqlite3"


def test_documents_dir(tmp_path):
    assert layout.documents_dir("alpha", _settings(tmp_path)) == tmp_path / "data" / "kbs" / "alpha" / "documents"


def test_parse_cache_dir(tmp_path):
    assert layout.parse_cache_dir(_settings(tmp_path)) == tmp_path / "data" / "cache" / "parse"


def test_jobs_db_path(tmp_path):
    assert layout.jobs_db_path(_settings(tmp_path)) == tmp_path / "data" / "jobs.sqlite3"


def test_bin_dir(tmp_path):
    assert layout.bin_dir(_settings(tmp_path)) == tmp_path / "data" / "bin"


def test_layout_derives_from_settings_env(tmp_path):
    assert layout.kbs_root() == tmp_path / "data" / "kbs"
    assert layout.bin_dir() == tmp_path / "data" / "bin"
    assert layout.jobs_db_path() == tmp_path / "data" / "jobs.sqlite3"


def test_ensure_data_dir_creates_all_roots(tmp_path):
    settings = _settings(tmp_path)
    layout.ensure_data_dir(settings)
    expected = [
        settings.data_dir,
        settings.data_dir / "kbs",
        settings.data_dir / "cache" / "parse",
        settings.data_dir / "bin",
        settings.qdrant_storage_dir,
    ]
    for path in expected:
        assert path.is_dir(), path
