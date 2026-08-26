import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import scholar_rag.core.operations as operations_module  # noqa: E402
import scholar_rag.ingest.pipeline as pipeline_module  # noqa: E402
from scholar_rag.core.errors import ConfirmTokenError, KbNotFoundError  # noqa: E402
from scholar_rag.core.jobs import JobManager  # noqa: E402
from scholar_rag.core.operations import Operations  # noqa: E402
from scholar_rag.store.catalog import Catalog  # noqa: E402
from scholar_rag.store.layout import catalog_path, documents_dir, jobs_db_path  # noqa: E402
from scholar_rag.store.vector_store import VectorStore  # noqa: E402
from tests.fixtures.gen_pdfs import PAPERS, to_markdown  # noqa: E402

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class FakeEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.sleep = 0.0
        self.intervals: list = []

    def embed_query(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dim - 1)

    def embed(self, texts: list[str]) -> list[list[float]]:
        start = time.monotonic()
        if self.sleep:
            time.sleep(self.sleep)
        self.intervals.append((start, time.monotonic()))
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def fake_models(monkeypatch):
    embedder = FakeEmbedder()
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(operations_module, "get_embedding_client", lambda: embedder)

    import scholar_rag.services.mineru as mineru_mod

    markdown_map = {name: to_markdown(blocks) for name, blocks in PAPERS.items()}

    def fake_parse(pdf_path: Path) -> str:
        stem = Path(pdf_path).stem
        if stem not in markdown_map:
            from scholar_rag.core.errors import PipelineStageError

            raise PipelineStageError(f"mineru failed for {stem}")
        return markdown_map[stem]

    monkeypatch.setattr(mineru_mod, "parse", fake_parse)
    return embedder


@pytest.fixture
def embedder(fake_models):
    return fake_models


@pytest.fixture
def ops(embedder):
    jobs = JobManager()
    instance = Operations(jobs)
    yield instance, jobs
    jobs.close()


@pytest.fixture
def kbr() -> str:
    return "kb" + uuid.uuid4().hex[:8]


def _wait_job(jobs: JobManager, job_id: str, statuses=("succeeded",), timeout: float = 120.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.get(job_id)
        if record.status in statuses:
            return record
        time.sleep(0.05)
    last = jobs.get(job_id)
    raise AssertionError(f"job {job_id} did not reach {statuses} (last: {last.status}, error={last.error})")


def _papers_dir(tmp_path: Path, names=("paper_a", "paper_b")) -> Path:
    import shutil

    folder = tmp_path / f"papers-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    for name in names:
        shutil.copy2(_FIXTURES / f"{name}.pdf", folder / f"{name}.pdf")
    return folder


def _create_kb(ops_jobs: tuple, folder: Path, kb: str) -> tuple:
    instance, jobs = ops_jobs
    job_id = instance.create_kb(kb, str(folder))
    record = _wait_job(jobs, job_id)
    return instance, jobs, record


def test_create_kb_ingests_all_papers_full_stack(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a", "paper_b"))
    job_id = instance.create_kb(kbr, str(folder))
    record = _wait_job(jobs, job_id)
    assert record.status == "succeeded"
    assert record.total == 2
    assert record.result["succeeded"] == 2
    assert record.result["failed"] == 0
    assert record.result["skipped"] == 0

    kb_dir = tmp_path / "data" / "kbs" / kbr
    assert kb_dir.is_dir()
    assert json.loads((kb_dir / "kb_meta.json").read_text(encoding="utf-8"))["dim"] == 8
    assert (kb_dir / "catalog.sqlite3").is_file()

    catalog = Catalog(catalog_path(kbr))
    assert catalog.count_documents() == 2
    _, rows = catalog.list_documents("title", 1, 10)
    doc_ids = {row["doc_id"] for row in rows}
    assert len(doc_ids) == 2
    for doc_id in doc_ids:
        doc_dir = documents_dir(kbr) / doc_id
        assert (doc_dir / "source.pdf").is_file()
        assert (doc_dir / "full_text.md").is_file()
        sections = json.loads((doc_dir / "sections.json").read_text(encoding="utf-8"))
        assert isinstance(sections["sections"], list)
        assert sections["total_chars"] > 0

    store = VectorStore.open(kbr)
    assert store.count() >= 2
    point_id, _score = store.query([1.0] + [0.0] * 7, None, limit=1)[0]
    payload = store.payloads([point_id])[point_id]
    assert payload["doc_id"] in doc_ids
    assert payload["title"]
    assert payload["title_norm"] == payload["title"].lower()
    assert payload["content"]
    assert isinstance(payload["added_ts"], float)


def test_create_kb_task_reports_single_failure(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a",))
    (folder / "bad_paper.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    job_id = instance.create_kb(kbr, str(folder))
    record = _wait_job(jobs, job_id)
    assert record.status == "succeeded"
    assert record.done == record.total == 2
    assert record.result["succeeded"] == 1
    assert record.result["failed"] == 1
    assert record.result["skipped"] == 0
    assert record.result["failures"][0]["doc"].endswith("bad_paper.pdf")
    assert "mineru failed" in record.result["failures"][0]["error"]
    assert Catalog(catalog_path(kbr)).count_documents() == 1


def test_add_document_single_document(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a",))
    _create_kb(ops, folder, kbr)
    catalog = Catalog(catalog_path(kbr))
    assert catalog.count_documents() == 1
    before = VectorStore.open(kbr).count()

    folder_b = _papers_dir(tmp_path, ("paper_b",))
    job_id = instance.add_document(kbr, str(folder_b / "paper_b.pdf"))
    record = _wait_job(jobs, job_id)
    assert record.status == "succeeded"
    assert record.total == 1
    assert record.result["doc_id"].startswith("doc-")
    assert catalog.count_documents() == 2
    assert VectorStore.open(kbr).count() > before
    doc_dir = documents_dir(kbr) / record.result["doc_id"]
    assert (doc_dir / "source.pdf").is_file()


def test_duplicate_add_raises_doc_exists_with_existing_id(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a",))
    _create_kb(ops, folder, kbr)
    catalog = Catalog(catalog_path(kbr))
    doc_id = catalog.list_documents("title", 1, 10)[1][0]["doc_id"]

    job_id = instance.add_document(kbr, str(folder / "paper_a.pdf"))
    record = _wait_job(jobs, job_id, statuses=("failed",))
    assert record.status == "failed"
    assert "already exists" in record.error
    assert doc_id in record.error
    assert catalog.count_documents() == 1


def test_resumed_create_kb_skips_existing_documents(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a", "paper_b"))
    _create_kb(ops, folder, kbr)
    before = VectorStore.open(kbr).count()

    job_id = jobs.submit("create_kb", kbr, {"folder": str(folder), "opts": {"skip_existing": True}})
    record = _wait_job(jobs, job_id)
    assert record.status == "succeeded"
    assert record.result["skipped"] == 2
    assert record.result["succeeded"] == 0
    assert record.result["failed"] == 0
    assert Catalog(catalog_path(kbr)).count_documents() == 2
    assert VectorStore.open(kbr).count() == before


def test_remove_document_cleans_qdrant_catalog_and_files(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a",))
    _create_kb(ops, folder, kbr)
    catalog = Catalog(catalog_path(kbr))
    doc_id = catalog.list_documents("title", 1, 10)[1][0]["doc_id"]
    store = VectorStore.open(kbr)

    result = instance.remove_document(kbr, doc_id)
    assert result["doc_id"] == doc_id
    assert result["chunks_deleted"] >= 1
    assert result["catalog_deleted"] is True
    assert result["files_deleted"] is True
    assert catalog.get_document(doc_id) is None
    assert store.count() == 0
    assert not (documents_dir(kbr) / doc_id).exists()


def test_delete_kb_three_state_lifecycle(ops, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a", "paper_b"))
    _create_kb(ops, folder, kbr)
    store = VectorStore.open(kbr)
    chunk_count = store.count()
    assert chunk_count >= 2

    preview = instance.delete_kb(kbr)
    assert preview["kb"] == kbr
    assert preview["doc_count"] == 2
    assert preview["chunk_count"] == chunk_count
    assert preview["expires_in"] == 600
    assert preview["created_at"]
    token = preview["confirm_token"]
    assert len(token) == 32

    with pytest.raises(ConfirmTokenError) as wrong:
        instance.delete_kb(kbr, "not-the-token")
    assert "mismatch" in str(wrong.value)
    assert (tmp_path / "data" / "kbs" / kbr).is_dir()

    instance._tokens[kbr] = (token, time.time() - 1)
    with pytest.raises(ConfirmTokenError) as expired:
        instance.delete_kb(kbr, token)
    assert "expired" in str(expired.value)
    assert (tmp_path / "data" / "kbs" / kbr).is_dir()

    fresh = instance.delete_kb(kbr)
    assert fresh["confirm_token"] != token
    result = instance.delete_kb(kbr, fresh["confirm_token"])
    assert result == {"kb": kbr, "deleted": True}

    assert not (tmp_path / "data" / "kbs" / kbr).exists()
    from scholar_rag.store.qdrant_manager import get_qdrant_manager

    assert not get_qdrant_manager().client().collection_exists(f"kb_{kbr}")
    remaining = sqlite3.connect(jobs_db_path()).execute(
        "SELECT COUNT(*) FROM jobs WHERE kb = ?", (kbr,)
    ).fetchone()[0]
    assert remaining == 0
    with pytest.raises(KbNotFoundError):
        instance.delete_kb(kbr)


def test_same_kb_concurrent_writes_are_serialized(ops, embedder, tmp_path, kbr):
    instance, jobs = ops
    folder = _papers_dir(tmp_path, ("paper_a",))
    _create_kb(ops, folder, kbr)
    embedder.intervals.clear()
    embedder.sleep = 0.3
    folder_bc = _papers_dir(tmp_path, ("paper_b", "paper_c"))

    first = instance.add_document(kbr, str(folder_bc / "paper_b.pdf"))
    second = instance.add_document(kbr, str(folder_bc / "paper_c.pdf"))
    _wait_job(jobs, first)
    _wait_job(jobs, second)

    assert len(embedder.intervals) == 2
    intervals = sorted(embedder.intervals)
    assert intervals[0][1] <= intervals[1][0]
    assert Catalog(catalog_path(kbr)).count_documents() == 3
