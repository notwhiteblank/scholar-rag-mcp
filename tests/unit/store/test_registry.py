import json
import os

import pytest

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import KbExistsError, KbNotFoundError
from scholar_rag.core.registry import Registry
from scholar_rag.core.types import MetadataResult
from scholar_rag.store.catalog import Catalog


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def registry(tmp_path):
    return Registry(Settings(data_dir=tmp_path / "data"))


def _meta(name: str) -> dict:
    return {
        "name": name,
        "created_at": "2026-08-26T00:00:00Z",
        "dim": 8,
        "embedding_model": "test-embed",
        "chunk": {"min": 100, "max": 500, "overlap": 50},
        "schema_version": 1,
    }


def _document(registry: Registry, name: str, doc_id: str) -> None:
    catalog = Catalog(registry.kb_path(name) / "catalog.sqlite3")
    meta = MetadataResult(
        title=doc_id,
        year=2024,
        journal="T",
        doi=None,
        authors=["A"],
        abstract=None,
        source="local",
    )
    catalog.upsert_document(meta, doc_id, f"/src/{doc_id}.pdf", doc_id)


VALID_NAMES = ["a", "abc", "a1", "a-b", "a_b", "9lives", "a" * 63]
INVALID_NAMES = ["", "-a", "_a", "A", "ab c", "a.b", "a/b", "a" * 65, "a b", "中"]


@pytest.mark.parametrize("name", VALID_NAMES)
def test_validate_name_accepts(registry, name: str):
    registry.validate_name(name)


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_validate_name_rejects(registry, name: str):
    with pytest.raises(ValueError):
        registry.validate_name(name)


def test_create_kb_creates_directory_layout(registry, tmp_path):
    registry.create_kb("alpha", _meta("alpha"))
    kb_dir = tmp_path / "data" / "kbs" / "alpha"
    assert kb_dir.is_dir()
    assert (kb_dir / "kb_meta.json").is_file()
    assert (kb_dir / "catalog.sqlite3").is_file()
    assert (kb_dir / "documents").is_dir()
    assert json.loads((kb_dir / "kb_meta.json").read_text()) == _meta("alpha")


def test_create_kb_duplicate_raises_kb_exists(registry):
    registry.create_kb("alpha", _meta("alpha"))
    with pytest.raises(KbExistsError):
        registry.create_kb("alpha", _meta("alpha"))
    assert len(registry.list_kbs()) == 1


def test_create_kb_invalid_name_raises_value_error(registry):
    with pytest.raises(ValueError):
        registry.create_kb("Invalid", _meta("Invalid"))


def test_kb_path(registry, tmp_path):
    path = registry.kb_path("alpha")
    assert path == tmp_path / "data" / "kbs" / "alpha"


def test_delete_kb(registry, tmp_path):
    registry.create_kb("alpha", _meta("alpha"))
    registry.delete_kb("alpha")
    assert not (tmp_path / "data" / "kbs" / "alpha").exists()
    assert registry.list_kbs() == []


def test_delete_missing_kb_raises_kb_not_found(registry):
    with pytest.raises(KbNotFoundError):
        registry.delete_kb("nope")


def test_list_kbs_empty_root(registry, tmp_path):
    assert registry.list_kbs() == []


def test_list_kbs_reports_ready_kbs(registry):
    registry.create_kb("alpha", _meta("alpha"))
    registry.create_kb("beta", _meta("beta"))
    _document(registry, "alpha", "doc1")
    _document(registry, "alpha", "doc2")
    kbs = {kb["name"]: kb for kb in registry.list_kbs()}
    assert set(kbs) == {"alpha", "beta"}
    assert kbs["alpha"]["doc_count"] == 2
    assert kbs["alpha"]["created_at"] == _meta("alpha")["created_at"]
    assert kbs["alpha"]["status"] == "ready"
    assert kbs["beta"]["doc_count"] == 0
    assert kbs["beta"]["status"] == "ready"


def test_list_kbs_marks_broken_kb(registry, tmp_path):
    registry.create_kb("alpha", _meta("alpha"))
    bad = tmp_path / "data" / "kbs" / "broken"
    bad.mkdir(parents=True)
    (bad / "kb_meta.json").write_text("{not json")
    kbs = {kb["name"]: kb for kb in registry.list_kbs()}
    assert kbs["alpha"]["status"] == "ready"
    assert kbs["broken"]["status"] == "broken"


def test_get_kb_lock_returns_shared_lock_per_name(registry):
    assert registry.get_kb_lock("alpha") is registry.get_kb_lock("alpha")
    assert registry.get_kb_lock("alpha") is not registry.get_kb_lock("beta")


def test_get_kb_lock_is_acquirable(registry):
    lock = registry.get_kb_lock("alpha")
    assert lock.acquire(timeout=1)
    lock.release()
