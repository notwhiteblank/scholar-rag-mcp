import json
import os
import time
from pathlib import Path

import pytest

import scholar_rag.core.operations as operations_module
from scholar_rag.core.config import Settings
from scholar_rag.core.errors import (
    ConfirmTokenError,
    DocNotFoundError,
    JobNotFoundError,
    KbExistsError,
    KbNotFoundError,
)
from scholar_rag.core.operations import Operations
from scholar_rag.core.registry import Registry
from scholar_rag.core.types import JobRecord, MetadataResult
from scholar_rag.store.catalog import Catalog


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0] * 8


class FakeVectorStore:
    created: list = []
    opened: list = []
    dropped: list = []
    counts: dict = {}

    @classmethod
    def create(cls, kb: str, dim: int):
        cls.created.append((kb, dim))
        return cls(kb)

    @classmethod
    def open(cls, kb: str):
        cls.opened.append(kb)
        return cls(kb)

    def __init__(self, kb: str) -> None:
        self.kb = kb

    def count(self) -> int:
        return FakeVectorStore.counts.get(self.kb, 0)

    def delete_doc(self, doc_id: str) -> int:
        return 3

    def drop(self) -> None:
        FakeVectorStore.dropped.append(self.kb)


class StubJobManager:
    def __init__(self) -> None:
        self.submitted: list = []
        self.handlers: dict = {}
        self.records: dict = {}
        self.deleted: list = []
        self.recover_calls = 0

    def submit(self, type: str, kb: str, params: dict) -> str:
        self.submitted.append((type, kb, dict(params)))
        job_id = f"job-{len(self.submitted)}"
        self.records[job_id] = JobRecord(
            job_id=job_id,
            type=type,
            kb=kb,
            status="pending",
            done=0,
            total=0,
            current_doc=None,
            error=None,
            result=None,
            created_at="t0",
            updated_at="t0",
        )
        return job_id

    def get(self, job_id) -> JobRecord:
        record = self.records.get(job_id)
        if record is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        return record

    def register_handler(self, type: str, handler) -> None:
        self.handlers[type] = handler

    def update_progress(self, job_id: str, done: int, current_doc: str) -> None:
        pass

    def set_total(self, job_id: str, total: int) -> None:
        pass

    def delete_by_kb(self, kb: str) -> int:
        self.deleted.append(kb)
        return 1

    def recover_interrupted(self) -> int:
        self.recover_calls += 1
        return 0


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCHOLAR_RAG_EMBED_MODEL", "test-embed")


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeVectorStore.created = []
    FakeVectorStore.opened = []
    FakeVectorStore.dropped = []
    FakeVectorStore.counts = {}


@pytest.fixture
def ops(monkeypatch):
    stub = StubJobManager()
    instance = Operations(stub)
    monkeypatch.setattr(operations_module, "get_embedding_client", lambda: FakeEmbedder())
    monkeypatch.setattr(operations_module, "VectorStore", FakeVectorStore)
    return instance, stub


def _make_kb(tmp_path: Path, name: str = "alphakb") -> None:
    registry = Registry(Settings(data_dir=tmp_path / "data"))
    registry.create_kb(
        name,
        {
            "name": name,
            "created_at": "2026-08-26T00:00:00Z",
            "dim": 8,
            "embedding_model": "test-embed",
            "chunk": {"min": 300, "max": 1500, "overlap": 100},
            "schema_version": 1,
        },
    )


def _fake_pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


def test_create_kb_invalid_name_raises_value_error(ops, tmp_path):
    instance, _ = ops
    with pytest.raises(ValueError):
        instance.create_kb("Invalid Name", str(tmp_path))


def test_create_kb_missing_folder_raises_value_error(ops, tmp_path):
    instance, _ = ops
    with pytest.raises(ValueError):
        instance.create_kb("valid", str(tmp_path / "nope"))


def test_create_kb_no_pdfs_raises_value_error(ops, tmp_path):
    instance, _ = ops
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError) as excinfo:
        instance.create_kb("valid", str(empty))
    assert "no pdf files found" in str(excinfo.value)


def test_create_kb_registers_kb_and_submits_job(ops, tmp_path):
    instance, stub = ops
    folder = tmp_path / "papers"
    (folder / "nested").mkdir(parents=True)
    _fake_pdf(folder, "a.pdf")
    _fake_pdf(folder / "nested", "b.PDF")
    job_id = instance.create_kb(
        "alphakb",
        str(folder),
        {"chunk_min": 200, "chunk_max": 900, "chunk_overlap": 60, "skip_existing": False},
    )
    assert job_id.startswith("job-")
    assert stub.submitted == [
        (
            "create_kb",
            "alphakb",
            {"folder": str(folder), "opts": {"chunk_min": 200, "chunk_max": 900, "chunk_overlap": 60, "skip_existing": False}},
        )
    ]
    assert FakeVectorStore.created == [("alphakb", 8)]
    meta_path = tmp_path / "data" / "kbs" / "alphakb" / "kb_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["name"] == "alphakb"
    assert meta["dim"] == 8
    assert meta["embedding_model"] == "test-embed"
    assert meta["chunk"] == {"min": 200, "max": 900, "overlap": 60}
    assert meta["schema_version"] == 1
    assert meta["created_at"]


def test_create_kb_duplicate_raises_kb_exists(ops, tmp_path):
    instance, _ = ops
    folder = tmp_path / "papers"
    folder.mkdir()
    _fake_pdf(folder, "a.pdf")
    instance.create_kb("alphakb", str(folder))
    with pytest.raises(KbExistsError):
        instance.create_kb("alphakb", str(folder))


def test_add_document_unknown_kb_raises_kb_not_found(ops, tmp_path):
    instance, _ = ops
    pdf = _fake_pdf(tmp_path)
    with pytest.raises(KbNotFoundError):
        instance.add_document("missing", str(pdf))


def test_add_document_missing_file_raises_value_error(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    with pytest.raises(ValueError):
        instance.add_document("alphakb", str(tmp_path / "nope.pdf"))


def test_add_document_submits_single_document_job(ops, tmp_path):
    instance, stub = ops
    _make_kb(tmp_path, "alphakb")
    pdf = _fake_pdf(tmp_path)
    job_id = instance.add_document("alphakb", str(pdf))
    assert job_id.startswith("job-")
    assert stub.submitted == [("add_document", "alphakb", {"pdf_path": str(pdf)})]


def test_remove_document_missing_doc_raises_doc_not_found(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    with pytest.raises(DocNotFoundError):
        instance.remove_document("alphakb", "nope")


def test_remove_document_deletes_qdrant_catalog_and_files(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    catalog = Catalog(tmp_path / "data" / "kbs" / "alphakb" / "catalog.sqlite3")
    catalog.add_document(
        MetadataResult(title="T", year=2020, journal=None, doi=None, authors=["A"], abstract=None, source="local"),
        "doc1",
        "/src/doc1.pdf",
        "hash1",
        ["point1", "point2"],
    )
    doc_dir = tmp_path / "data" / "kbs" / "alphakb" / "documents" / "doc1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "source.pdf").write_bytes(b"x")
    result = instance.remove_document("alphakb", "doc1")
    assert result["doc_id"] == "doc1"
    assert result["chunks_deleted"] == 3
    assert result["catalog_deleted"] is True
    assert result["files_deleted"] is True
    assert catalog.get_document("doc1") is None
    assert not doc_dir.exists()


def test_delete_kb_unknown_kb_raises_kb_not_found(ops, tmp_path):
    instance, _ = ops
    with pytest.raises(KbNotFoundError):
        instance.delete_kb("missing")


def test_delete_kb_preview_returns_counts_and_token(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    FakeVectorStore.counts["alphakb"] = 42
    result = instance.delete_kb("alphakb")
    assert result["kb"] == "alphakb"
    assert result["doc_count"] == 0
    assert result["chunk_count"] == 42
    assert result["created_at"] == "2026-08-26T00:00:00Z"
    assert result["expires_in"] == 600
    assert len(result["confirm_token"]) == 32
    token, expires_at = instance._tokens["alphakb"]
    assert token == result["confirm_token"]
    assert expires_at > time.time()


def test_delete_kb_wrong_token_raises(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    instance.delete_kb("alphakb")
    with pytest.raises(ConfirmTokenError) as excinfo:
        instance.delete_kb("alphakb", "wrongtoken")
    assert "mismatch" in str(excinfo.value)


def test_delete_kb_unknown_token_raises(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    with pytest.raises(ConfirmTokenError) as excinfo:
        instance.delete_kb("alphakb", "sometoken")
    assert "no confirm token" in str(excinfo.value)


def test_delete_kb_expired_token_raises(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    preview = instance.delete_kb("alphakb")
    instance._tokens["alphakb"] = (preview["confirm_token"], time.time() - 1)
    with pytest.raises(ConfirmTokenError) as excinfo:
        instance.delete_kb("alphakb", preview["confirm_token"])
    assert "expired" in str(excinfo.value)


def test_delete_kb_with_valid_token_deletes_all(ops, tmp_path):
    instance, stub = ops
    _make_kb(tmp_path, "alphakb")
    fake_pdf = _fake_pdf(tmp_path, "notused.pdf")
    instance.add_document("alphakb", str(fake_pdf))
    preview = instance.delete_kb("alphakb")
    result = instance.delete_kb("alphakb", preview["confirm_token"])
    assert result == {"kb": "alphakb", "deleted": True}
    assert FakeVectorStore.dropped == ["alphakb"]
    assert stub.deleted == ["alphakb"]
    assert not (tmp_path / "data" / "kbs" / "alphakb").exists()
    assert "alphakb" not in instance._tokens


def test_delete_kb_clears_token_after_success(ops, tmp_path):
    instance, _ = ops
    _make_kb(tmp_path, "alphakb")
    preview = instance.delete_kb("alphakb")
    assert len(instance._tokens) == 1
    instance.delete_kb("alphakb", preview["confirm_token"])
    assert instance._tokens == {}
    with pytest.raises(KbNotFoundError):
        instance.delete_kb("alphakb", preview["confirm_token"])


def test_list_kbs_and_get_job_delegate(ops, tmp_path):
    instance, stub = ops
    _make_kb(tmp_path, "alphakb")
    kbs = instance.list_kbs()
    assert [kb["name"] for kb in kbs] == ["alphakb"]
    job_id = instance.add_document("alphakb", str(_fake_pdf(tmp_path)))
    assert instance.get_job(job_id).status == "pending"
    with pytest.raises(JobNotFoundError):
        instance.get_job("nope")


def test_startup_recovers_interrupted(ops):
    instance, stub = ops
    instance.startup()
    assert stub.recover_calls == 1
