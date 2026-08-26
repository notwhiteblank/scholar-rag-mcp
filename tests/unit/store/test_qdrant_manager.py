import os

import httpx
import pytest
from qdrant_client import QdrantClient

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.store import qdrant_manager
from scholar_rag.store.qdrant_manager import QdrantManager, get_qdrant_manager, reset_qdrant_manager


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    reset_qdrant_manager()


def _settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_storage_dir=tmp_path / "qdrant",
    )


def test_direct_connect_when_url_configured(tmp_path):
    manager = QdrantManager(_settings(tmp_path))
    client = manager.client()
    assert isinstance(client, QdrantClient)
    assert manager._process is None


def test_singleton_and_reset():
    assert get_qdrant_manager() is get_qdrant_manager()
    first = get_qdrant_manager()
    reset_qdrant_manager()
    assert get_qdrant_manager() is not first


def test_missing_qdrant_bin_raises_service_unavailable(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        qdrant_url="",
        qdrant_bin="/nonexistent/qdrant",
        qdrant_storage_dir=tmp_path / "qdrant",
    )
    with pytest.raises(ServiceUnavailableError):
        QdrantManager(settings).client()


def test_download_failure_raises_service_unavailable_with_url(monkeypatch, tmp_path):
    def _stream(*args, **kwargs):
        raise httpx.ConnectError("mock network failure", request=None)

    monkeypatch.setattr(httpx, "stream", _stream)
    settings = Settings(
        data_dir=tmp_path / "data",
        qdrant_url="",
        qdrant_storage_dir=tmp_path / "qdrant",
    )
    with pytest.raises(ServiceUnavailableError) as excinfo:
        QdrantManager(settings).client()
    assert "github.com" in str(excinfo.value)


def test_version_pinned_to_1_12_5():
    assert qdrant_manager.QDRANT_VERSION == "1.12.5"
