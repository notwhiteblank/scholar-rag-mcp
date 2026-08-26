import os
import re
import uuid
from pathlib import Path

import pytest

import scholar_rag.ingest.pipeline as pipeline_module
from scholar_rag.core.errors import DocExistsError, PipelineStageError
from scholar_rag.core.types import Chunk, MetadataResult

MARKDOWN = (
    "# Pipeline Test Paper\n\n"
    "Alice Doe and Bob Smith\n\n"
    "2021\n\n"
    "## Abstract\n\n"
    "A synthetic abstract body. " * 20
    + "\n\n## Introduction\n\nIntro body text. " * 10
)

META = MetadataResult(
    title="Pipeline Test Paper",
    year=2021,
    journal="Journal of Tests",
    doi="10.1/x",
    authors=["Alice Doe", "Bob Smith"],
    abstract="An abstract.",
    source="local",
    keywords=("k1",),
)


class FakeEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.texts: list[list[str]] = []

    def embed_query(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (self.dim - 1)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


class FakeVectorStore:
    registry: dict = {}
    last: "FakeVectorStore | None" = None
    upsert_count = 0

    @classmethod
    def open(cls, kb: str) -> "FakeVectorStore":
        store = cls.registry.setdefault(kb, cls(kb))
        cls.last = store
        return store

    def __init__(self, kb: str) -> None:
        self.kb = kb
        self.points: dict = {}
        self.events: list = []

    def upsert_chunks(self, doc_id, chunks, embeds, payload_base):
        FakeVectorStore.upsert_count += 1
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{chunk.chunk_index}")) for chunk in chunks
        ]
        for point_id, chunk in zip(point_ids, chunks, strict=True):
            self.points[point_id] = {
                **payload_base,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
        self.events.append(("qdrant", doc_id, point_ids, payload_base))
        return point_ids

    def delete_doc(self, doc_id):
        left = {p: v for p, v in self.points.items() if v["doc_id"] != doc_id}
        deleted = len(self.points) - len(left)
        self.points = left
        self.events.append(("delete", doc_id))
        return deleted


class FakeCatalog:
    docs: dict = {}
    add_calls = 0
    last: "FakeCatalog | None" = None

    def __init__(self, path) -> None:
        FakeCatalog.last = self
        self.path = path

    def find_by_hash(self, content_hash):
        for entry in FakeCatalog.docs.values():
            if entry["content_hash"] == content_hash:
                return entry
        return None

    def add_document(self, meta, doc_id, source_path, content_hash, chunk_ids):
        FakeCatalog.add_calls += 1
        FakeCatalog.docs[doc_id] = {
            "doc_id": doc_id,
            "title": meta.title,
            "content_hash": content_hash,
            "chunk_ids": list(chunk_ids),
        }

    def delete_document(self, doc_id):
        FakeCatalog.docs.pop(doc_id, None)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "false")
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "false")
    monkeypatch.setenv("SCHOLAR_RAG_CROSSREF_ENABLED", "false")


@pytest.fixture(autouse=True)
def _reset_fakes():
    FakeVectorStore.registry = {}
    FakeVectorStore.last = None
    FakeVectorStore.upsert_count = 0
    FakeCatalog.docs = {}
    FakeCatalog.add_calls = 0
    FakeCatalog.last = None


@pytest.fixture
def harness(monkeypatch):
    import scholar_rag.services.mineru as mineru_mod

    def fake_parse(pdf_path: Path) -> str:
        return MARKDOWN

    monkeypatch.setattr(mineru_mod, "parse", fake_parse)
    monkeypatch.setattr(pipeline_module, "stage_metadata", lambda staged, md: META)

    def fake_chunks(annotated, meta, cfg):
        return [
            Chunk(
                chunk_index=0,
                content="first chunk content",
                section="Abstract",
                heading_path="Abstract",
                metadata={"title": meta.title},
            ),
            Chunk(
                chunk_index=1,
                content="second chunk content",
                section="Abstract",
                heading_path="Abstract",
                metadata={"title": meta.title},
            ),
        ]

    monkeypatch.setattr(pipeline_module, "stage_chunks", fake_chunks)
    monkeypatch.setattr(pipeline_module, "get_embedding_client", lambda: FakeEmbedder())
    monkeypatch.setattr(pipeline_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(pipeline_module, "Catalog", FakeCatalog)


def _pdf(tmp_path: Path, name: str = "paper.pdf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


def _disk_events(monkeypatch):
    recorded: list[str] = []
    real = pipeline_module._persist_disk

    def wrapper(*args):
        recorded.append("disk")
        return real(*args)

    monkeypatch.setattr(pipeline_module, "_persist_disk", wrapper)
    return recorded


def test_ingest_document_returns_summary_with_doc_id(harness, tmp_path):
    outcome = pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    assert outcome["doc_id"].startswith("doc-")
    assert outcome["title"] == "Pipeline Test Paper"
    assert outcome["chunks"] == 2
    assert outcome["skipped"] is False


def test_persist_order_is_qdrant_then_catalog_then_disk(harness, tmp_path, monkeypatch):
    disk = _disk_events(monkeypatch)
    pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    store = FakeVectorStore.registry["kbtest"]
    assert store.events[0][0] == "qdrant"
    assert FakeCatalog.last.add_calls == 1
    assert disk == ["disk"]


def test_payload_base_contains_document_fields(harness, tmp_path):
    pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    store = FakeVectorStore.registry["kbtest"]
    point_id, payload = next(iter(store.points.items()))
    assert payload["doc_id"] == store.events[0][1]
    assert payload["title"] == "Pipeline Test Paper"
    assert payload["title_norm"] == "pipeline test paper"
    assert payload["year"] == 2021
    assert payload["journal"] == "Journal of Tests"
    assert payload["journal_norm"] == "journal of tests"
    assert payload["first_author"] == "Alice Doe"
    assert payload["first_author_norm"] == "alice doe"
    assert payload["author_count"] == 2
    assert payload["doi"] == "10.1/x"
    assert payload["source_file"].endswith("paper.pdf")
    assert payload["content_hash"]
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", payload["added_at"])
    assert isinstance(payload["added_ts"], float)
    assert payload["chunk_index"] == 0
    assert payload["content"] == "first chunk content"


def test_catalog_chunk_ids_match_qdrant_point_ids(harness, tmp_path):
    pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    store = FakeVectorStore.registry["kbtest"]
    _, doc_id, point_ids, _ = store.events[0]
    assert FakeCatalog.last.docs[doc_id]["chunk_ids"] == point_ids


def test_disk_files_written_atomically(harness, tmp_path):
    pdf = _pdf(tmp_path)
    outcome = pipeline_module.ingest_document(pdf, "kbtest", pipeline_module.ChunkConfig())
    doc_dir = tmp_path / "data" / "kbs" / "kbtest" / "documents" / outcome["doc_id"]
    assert (doc_dir / "source.pdf").is_file()
    assert (doc_dir / "full_text.md").is_file()
    sections = __import__("json").loads((doc_dir / "sections.json").read_text(encoding="utf-8"))
    assert isinstance(sections["sections"], list)
    assert sections["total_chars"] >= 0
    assert isinstance(sections["section_texts"], dict)
    assert not [p for p in doc_dir.parent.iterdir() if p.name.startswith(".ingest-")]


def test_sections_json_section_texts_match_outline_counts(harness, tmp_path):
    pdf = _pdf(tmp_path)
    outcome = pipeline_module.ingest_document(pdf, "kbtest", pipeline_module.ChunkConfig())
    doc_dir = tmp_path / "data" / "kbs" / "kbtest" / "documents" / outcome["doc_id"]
    sections = __import__("json").loads((doc_dir / "sections.json").read_text(encoding="utf-8"))
    names = [pair[0] for pair in sections["sections"]]
    counts = {pair[0]: int(pair[1]) for pair in sections["sections"]}
    assert set(sections["section_texts"]) == set(names)
    for name in names:
        assert len(sections["section_texts"][name]) == counts[name]
    assert sum(len(text) for text in sections["section_texts"].values()) == sections["total_chars"]


def test_catalog_persist_failure_cleans_qdrant_points(harness, tmp_path, monkeypatch):
    def boom(self, *args):
        raise RuntimeError("catalog write failed")

    monkeypatch.setattr(pipeline_module.Catalog, "add_document", boom)
    with pytest.raises(PipelineStageError) as excinfo:
        pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    assert excinfo.value.stage == "persist_catalog"
    assert "catalog write failed" in excinfo.value.cause
    store = FakeVectorStore.registry["kbtest"]
    assert store.points == {}
    assert ("delete", (store.events[0][1])) in [tuple(e[:2]) for e in store.events]
    assert not [p for p in (tmp_path / "data" / "kbs" / "kbtest" / "documents").iterdir() if p.name.startswith(".ingest-")]


def test_disk_persist_failure_cleans_qdrant_and_catalog(harness, tmp_path, monkeypatch):
    def boom(*args):
        raise RuntimeError("disk write failed")

    monkeypatch.setattr(pipeline_module, "_persist_disk", boom)
    with pytest.raises(PipelineStageError) as excinfo:
        pipeline_module.ingest_document(_pdf(tmp_path), "kbtest", pipeline_module.ChunkConfig())
    assert excinfo.value.stage == "persist_disk"
    assert FakeVectorStore.registry["kbtest"].points == {}
    assert FakeCatalog.last.docs == {}


def test_duplicate_add_raises_doc_exists_with_existing_id(harness, tmp_path):
    pdf = _pdf(tmp_path)
    first = pipeline_module.ingest_document(pdf, "kbtest", pipeline_module.ChunkConfig())
    with pytest.raises(DocExistsError) as excinfo:
        pipeline_module.ingest_document(pdf, "kbtest", pipeline_module.ChunkConfig())
    assert excinfo.value.doc_id == first["doc_id"]
    assert "already exists" in str(excinfo.value)


def test_skip_existing_returns_skipped_with_existing_doc_id(harness, tmp_path):
    pdf = _pdf(tmp_path)
    first = pipeline_module.ingest_document(pdf, "kbtest", pipeline_module.ChunkConfig())
    outcome = pipeline_module.ingest_document(
        pdf, "kbtest", pipeline_module.ChunkConfig(), skip_existing=True
    )
    assert outcome["skipped"] is True
    assert outcome["doc_id"] == first["doc_id"]
    assert outcome["chunks"] == 0
    assert FakeVectorStore.upsert_count == 1
    store = FakeVectorStore.registry["kbtest"]
    assert not [e for e in store.events if e[0] == "delete"]


def test_skip_existing_cleans_up_staging_dir(harness, tmp_path):
    pdf = _pdf(tmp_path)
    first = pipeline_module.ingest_document(
        pdf, "kbtest", pipeline_module.ChunkConfig(), skip_existing=True
    )
    second = pipeline_module.ingest_document(
        pdf, "kbtest", pipeline_module.ChunkConfig(), skip_existing=True
    )
    assert first["skipped"] is False
    assert second["skipped"] is True
    assert second["doc_id"] == first["doc_id"]
    assert second["title"] == first["title"]
    documents_area = tmp_path / "data" / "kbs" / "kbtest" / "documents"
    assert [p for p in documents_area.iterdir() if p.name.startswith(".ingest-")] == []
    stored = documents_area / first["doc_id"]
    assert len(list(stored.rglob("*.pdf"))) == 1
