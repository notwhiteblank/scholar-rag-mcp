import json
import math
import os
import time
import uuid

import pytest

import scholar_rag.core.operations as operations_module
import scholar_rag.server.tools as tools_module
from scholar_rag.core.jobs import JobManager
from scholar_rag.core.operations import Operations
from scholar_rag.core.registry import Registry
from scholar_rag.core.types import MetadataResult, RetrievedChunk
from scholar_rag.retrieve.engine import SearchOutcome
from scholar_rag.store.catalog import Catalog
from scholar_rag.store.layout import catalog_path, documents_dir

ABSTRACT = "Abstract of the paper."
BODY = "B" * 12000
FULL_TEXT = ABSTRACT + "\n\n" + BODY


def _sections_payload(body: str, abstract: str = ABSTRACT) -> dict:
    return {
        "sections": [["Abstract", len(abstract)], ["Body", len(body)]],
        "total_chars": len(abstract) + len(body),
        "section_texts": {"Abstract": abstract, "Body": body},
    }


class FakeEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dim - 1)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dim for _ in texts]


class FakeVectorStore:
    @classmethod
    def create(cls, kb: str, dim: int) -> "FakeVectorStore":
        return cls()

    @classmethod
    def open(cls, kb: str) -> "FakeVectorStore":
        return cls()

    def count(self) -> int:
        return 7

    def delete_doc(self, doc_id: str) -> int:
        return 3

    def drop(self) -> None:
        return None

    def upsert_chunks(self, doc_id, chunks, embeds, payload_base):  # type: ignore[no-untyped-def]
        return ["point-1"]


class FakeEngine:
    def __init__(self, kb: str) -> None:
        self.kb = kb

    def search(  # type: ignore[no-untyped-def]
        self, query, top_k=10, min_score=None, metadata_filter=None
    ) -> SearchOutcome:
        return SearchOutcome(
            chunks=[
                RetrievedChunk(
                    content="chunk body",
                    metadata={"doc_id": "doc-1", "section": "Body", "year": 2021},
                    embed_score=0.5,
                    rerank_score=0.9,
                )
            ],
            timings={"embed_ms": 1.0, "search_ms": 2.0, "rerank_ms": 3.0},
            warning=None,
        )


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "false")
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "false")
    monkeypatch.setenv("SCHOLAR_RAG_CROSSREF_ENABLED", "false")
    return tmp_path


@pytest.fixture
def ops(env, monkeypatch):
    monkeypatch.setattr(operations_module, "get_embedding_client", lambda: FakeEmbedder())
    monkeypatch.setattr(operations_module, "VectorStore", FakeVectorStore)
    jobs = JobManager()
    instance = Operations(jobs)
    monkeypatch.setattr(tools_module, "_operations", lambda: instance)
    yield instance
    jobs.close()


def _kb_name() -> str:
    return "kb" + uuid.uuid4().hex[:8]


def _seed_kb(env, name: str) -> None:
    Registry().create_kb(
        name,
        {
            "name": name,
            "created_at": "2024-01-01T00:00:00Z",
            "dim": 8,
            "embedding_model": "fake",
            "chunk": {"min": 300, "max": 1500, "overlap": 100},
            "schema_version": 1,
        },
    )


def _seed_doc(
    env,
    kb: str,
    doc_id: str = "doc-1",
    title: str = "Title One",
    year: int = 2021,
    body: str = FULL_TEXT,
) -> str:
    meta = MetadataResult(
        title=title,
        year=year,
        journal="Journal X",
        doi="10.1/x",
        authors=["Alice Doe"],
        abstract=ABSTRACT,
        source="local",
        keywords=("key",),
    )
    catalog = Catalog(catalog_path(kb))
    catalog.add_document(meta, doc_id, str(env / "a.pdf"), "hash-" + doc_id, ["point-" + doc_id])
    doc_dir = documents_dir(kb) / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "full_text.md").write_text(body, encoding="utf-8")
    (doc_dir / "sections.json").write_text(
        json.dumps(_sections_payload(body)), encoding="utf-8"
    )
    return doc_id


def _wait_status(jobs: JobManager, job_id: str, status: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if jobs.get(job_id).status == status:
            return
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {status}: {jobs.get(job_id).status}")


def test_unknown_tool_returns_structured_error():
    result = tools_module.dispatch_tool("no_such_tool", {})
    assert result["error_code"] == "unknown_tool"
    assert "message" in result
    assert "hint" in result


def test_arguments_must_be_object():
    result = tools_module.dispatch_tool("list_kbs", None)  # type: ignore[arg-type]
    assert result["error_code"] == "invalid_request"


def test_create_kb_returns_job_id(env, ops, tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = tools_module.dispatch_tool(
        "create_kb",
        {"kb_name": "newkb", "folder_path": str(folder), "skip_existing": True},
    )
    assert set(result) == {"job_id"}
    assert len(result["job_id"]) == 32


def test_create_kb_invalid_name_is_invalid_request(env, ops, tmp_path):
    folder = tmp_path / "papers"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = tools_module.dispatch_tool(
        "create_kb", {"kb_name": "Bad Name", "folder_path": str(folder)}
    )
    assert result["error_code"] == "invalid_request"


def test_create_kb_missing_folder_is_invalid_request(env, ops, tmp_path):
    result = tools_module.dispatch_tool(
        "create_kb", {"kb_name": "newkb", "folder_path": str(tmp_path / "nope")}
    )
    assert result["error_code"] == "invalid_request"


def test_add_document_returns_job_id(env, ops, tmp_path):
    kb = _kb_name()
    _seed_kb(env, kb)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = tools_module.dispatch_tool(
        "add_document", {"kb": kb, "pdf_path": str(pdf)}
    )
    assert set(result) == {"job_id"}
    assert len(result["job_id"]) == 32


def test_add_document_missing_file_is_invalid_request(env, ops, tmp_path):
    kb = _kb_name()
    _seed_kb(env, kb)
    result = tools_module.dispatch_tool(
        "add_document", {"kb": kb, "pdf_path": str(tmp_path / "nope.pdf")}
    )
    assert result["error_code"] == "invalid_request"


def test_list_kbs_roundtrip(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    result = tools_module.dispatch_tool("list_kbs", {})
    entries = result["kbs"]
    assert [entry["name"] for entry in entries] == [kb]
    assert entries[0]["doc_count"] == 1
    assert entries[0]["status"] == "ready"
    assert entries[0]["created_at"] == "2024-01-01T00:00:00Z"
    assert entries[0]["chunk_count"] == 7


def test_list_kbs_broken_kb_reports_zero_chunks(env, ops):
    kb = _kb_name()
    (env / "data" / "kbs" / kb).mkdir(parents=True)
    (env / "data" / "kbs" / kb / "kb_meta.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    result = tools_module.dispatch_tool("list_kbs", {})
    entries = result["kbs"]
    assert [entry["name"] for entry in entries] == [kb]
    assert entries[0]["status"] == "broken"
    assert entries[0]["chunk_count"] == 0


def test_list_kbs_empty(env, ops):
    result = tools_module.dispatch_tool("list_kbs", {})
    assert result == {"kbs": []}


def test_search_documents_filters_and_pages(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb, doc_id="doc-a", title="Graph Neural Networks", year=2020)
    _seed_doc(env, kb, doc_id="doc-b", title="Attention Mechanisms", year=2021)
    _seed_doc(env, kb, doc_id="doc-c", title="Graph Embeddings", year=2022)
    result = tools_module.dispatch_tool(
        "search_documents", {"kb": kb, "title": "graph", "page": 1, "page_size": 1}
    )
    assert result["total"] == 2
    assert result["page"] == 1
    assert result["page_size"] == 1
    assert len(result["results"]) == 1
    assert set(result["results"][0]) == {
        "doc_id",
        "title",
        "year",
        "journal",
        "first_author",
        "doi",
    }
    assert result["results"][0]["title"] == "Graph Neural Networks"


def test_search_documents_page_size_cap(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    result = tools_module.dispatch_tool(
        "search_documents", {"kb": kb, "page_size": 21}
    )
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool(
        "search_documents", {"kb": kb, "page": 0}
    )
    assert result["error_code"] == "invalid_request"


def test_list_documents_sort_and_caps(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb, doc_id="doc-a", title="Zed", year=2020)
    _seed_doc(env, kb, doc_id="doc-b", title="Alpha", year=2021)
    result = tools_module.dispatch_tool(
        "list_documents", {"kb": kb, "sort": "title", "page": 1, "page_size": 1}
    )
    assert result["total"] == 2
    assert [row["title"] for row in result["results"]] == ["Alpha"]
    result = tools_module.dispatch_tool("list_documents", {"kb": kb, "page_size": 21})
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool("list_documents", {"kb": kb, "page": 0})
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool("list_documents", {"kb": kb, "sort": "bogus"})
    assert result["error_code"] == "invalid_request"


def test_get_document_overview(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    result = tools_module.dispatch_tool("get_document", {"kb": kb, "doc_id": "doc-1"})
    assert result["doc_id"] == "doc-1"
    assert result["metadata"]["title"] == "Title One"
    assert result["abstract"] == ABSTRACT
    assert result["outline"] == [
        {"section": "Abstract", "char_count": len(ABSTRACT)},
        {"section": "Body", "char_count": len(FULL_TEXT)},
    ]
    assert result["total_chars"] == len(ABSTRACT) + len(FULL_TEXT)
    assert "get_document_text" in result["usage_hint"]


def test_get_document_text_full_text_pagination_arithmetic(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    total_chars = len(FULL_TEXT)
    page_size = 6000
    total_pages = math.ceil(total_chars / page_size)
    reassembled: list[str] = []
    for page in range(1, total_pages + 1):
        result = tools_module.dispatch_tool(
            "get_document_text",
            {"kb": kb, "doc_id": "doc-1", "page": page, "page_size": page_size},
        )
        assert result["page"] == page
        assert result["total_pages"] == total_pages
        assert result["total_chars"] == total_chars
        resumed = "next page: " + str(page + 1) if page < total_pages else None
        assert result["next_hint"] == resumed
        reassembled.append(result["content"])
    assert "".join(reassembled) == FULL_TEXT
    assert all(len(part) <= page_size for part in reassembled)


def test_get_document_text_page_size_cap(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    result = tools_module.dispatch_tool(
        "get_document_text", {"kb": kb, "doc_id": "doc-1", "page_size": 8001}
    )
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool(
        "get_document_text", {"kb": kb, "doc_id": "doc-1", "page_size": 0}
    )
    assert result["error_code"] == "invalid_request"


def test_get_document_text_page_out_of_range(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    total_pages = math.ceil(len(FULL_TEXT) / 6000)
    result = tools_module.dispatch_tool(
        "get_document_text",
        {"kb": kb, "doc_id": "doc-1", "page": total_pages + 1, "page_size": 6000},
    )
    assert result["error_code"] == "invalid_request"
    assert f"pages 1..{total_pages}" in result["message"]


def test_get_document_text_by_section(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    result = tools_module.dispatch_tool(
        "get_document_text",
        {"kb": kb, "doc_id": "doc-1", "section": "Body", "page_size": 5000},
    )
    assert result["total_chars"] == len(FULL_TEXT)
    assert result["total_pages"] == math.ceil(len(FULL_TEXT) / 5000)
    assert result["content"] == FULL_TEXT[:5000]


def test_get_document_text_missing_section_lists_available(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    result = tools_module.dispatch_tool(
        "get_document_text",
        {"kb": kb, "doc_id": "doc-1", "section": "Nonexistent"},
    )
    assert result["error_code"] == "invalid_request"
    assert "Abstract" in result["message"]
    assert "Body" in result["message"]


def test_search_chunks_structure(env, ops, monkeypatch):
    kb = _kb_name()
    _seed_kb(env, kb)
    monkeypatch.setattr(tools_module, "RetrievalEngine", FakeEngine)
    result = tools_module.dispatch_tool(
        "search_chunks", {"kb": kb, "query": "attention"}
    )
    assert result["warning"] is None
    assert result["timings"] == {"embed_ms": 1.0, "search_ms": 2.0, "rerank_ms": 3.0}
    assert result["chunks"] == [
        {
            "content": "chunk body",
            "metadata": {"doc_id": "doc-1", "section": "Body", "year": 2021},
            "embed_score": 0.5,
            "rerank_score": 0.9,
        }
    ]


def test_search_chunks_caps(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    result = tools_module.dispatch_tool(
        "search_chunks", {"kb": kb, "query": "q", "top_k": 51}
    )
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool(
        "search_chunks", {"kb": kb, "query": "q", "top_k": 0}
    )
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool(
        "search_chunks", {"kb": kb, "query": "q", "min_score": 1.5}
    )
    assert result["error_code"] == "invalid_request"


def test_remove_document_success_and_missing(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    result = tools_module.dispatch_tool(
        "remove_document", {"kb": kb, "doc_id": "doc-1"}
    )
    assert result == {
        "doc_id": "doc-1",
        "chunks_deleted": 3,
        "catalog_deleted": True,
        "files_deleted": True,
    }
    result = tools_module.dispatch_tool(
        "remove_document", {"kb": kb, "doc_id": "nope"}
    )
    assert result["error_code"] == "doc_not_found"


def test_delete_kb_two_phase(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb)
    preview = tools_module.dispatch_tool("delete_kb", {"kb": kb})
    assert preview["kb"] == kb
    assert preview["doc_count"] == 1
    assert preview["chunk_count"] == 7
    assert preview["expires_in"] == 600
    assert len(preview["confirm_token"]) == 32
    assert "deleted" not in preview
    rejected = tools_module.dispatch_tool(
        "delete_kb", {"kb": kb, "confirm_token": "wrong-token"}
    )
    assert rejected["error_code"] == "confirm_token_invalid"
    confirmed = tools_module.dispatch_tool(
        "delete_kb", {"kb": kb, "confirm_token": preview["confirm_token"]}
    )
    assert confirmed == {"kb": kb, "deleted": True}


def test_get_job_success_and_missing(env, ops):
    kb = _kb_name()
    ops._jobs.register_handler("succeed", lambda job_id, params: {"ok": True})
    job_id = ops._jobs.submit("succeed", kb, {})
    _wait_status(ops._jobs, job_id, "succeeded")
    result = tools_module.dispatch_tool("get_job", {"job_id": job_id})
    assert result["status"] == "succeeded"
    assert result["type"] == "succeed"
    assert result["kb"] == kb
    assert result["progress"] == {"done": 0, "total": 0}
    assert result["result_summary"] == {"ok": True}
    assert set(result["timings"]) == {"elapsed_s"}
    assert result["timings"]["elapsed_s"] >= 0.0
    missing = tools_module.dispatch_tool("get_job", {"job_id": "no-such"})
    assert missing["error_code"] == "job_not_found"


def test_unknown_kb_is_kb_not_found(env, ops):
    assert tools_module.dispatch_tool(
        "search_chunks", {"kb": "ghost", "query": "q"}
    )["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("search_documents", {"kb": "ghost"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("list_documents", {"kb": "ghost"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("get_document", {"kb": "ghost", "doc_id": "d"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("get_document_text", {"kb": "ghost", "doc_id": "d"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("add_document", {"kb": "ghost", "pdf_path": "x.pdf"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("remove_document", {"kb": "ghost", "doc_id": "d"})["error_code"] == "kb_not_found"
    assert tools_module.dispatch_tool("delete_kb", {"kb": "ghost"})["error_code"] == "kb_not_found"


def test_unknown_document_is_doc_not_found(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    assert tools_module.dispatch_tool("get_document", {"kb": kb, "doc_id": "nope"})[
        "error_code"
    ] == "doc_not_found"
    assert tools_module.dispatch_tool("get_document_text", {"kb": kb, "doc_id": "nope"})[
        "error_code"
    ] == "doc_not_found"


def test_invalid_arguments_type_or_shape(env, ops):
    result = tools_module.dispatch_tool("get_document", {"kb": 42, "doc_id": "d"})
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool("search_chunks", {"kb": "k"})
    assert result["error_code"] == "invalid_request"
    result = tools_module.dispatch_tool("get_job", {})
    assert result["error_code"] == "invalid_request"


def test_internal_error_does_not_leak_stack(env, ops, monkeypatch):
    kb = _kb_name()
    _seed_kb(env, kb)
    monkeypatch.setattr(tools_module, "RetrievalEngine", FakeEngine)

    def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(FakeEngine, "search", boom)
    result = tools_module.dispatch_tool("search_chunks", {"kb": kb, "query": "q"})
    assert result["error_code"] == "internal_error"
    assert "secret internal detail" not in result["message"]


def test_tool_functions_delegate_to_dispatch(env, ops):
    kb = _kb_name()
    _seed_kb(env, kb)
    _seed_doc(env, kb, body=BODY)
    result = tools_module.tool_get_document(kb=kb, doc_id="doc-1")
    assert result["doc_id"] == "doc-1"
    assert result["outline"][0]["section"] == "Abstract"
