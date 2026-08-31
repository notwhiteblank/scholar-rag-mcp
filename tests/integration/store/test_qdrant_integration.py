import socket
import tarfile
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import httpx
import pytest
from qdrant_client import models

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.core.types import Chunk
from scholar_rag.store.qdrant_manager import get_qdrant_manager, reset_qdrant_manager
from scholar_rag.store.vector_store import VectorStore

pytestmark = pytest.mark.integration


def test_server_reports_qdrant_1_12_5(qdrant_instance):
    response = httpx.get(qdrant_instance["url"] + "/", timeout=5)
    assert response.status_code == 200
    assert response.json()["version"] == "1.12.5"


def _chunks(doc_id: str, count: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_index=index,
            content=f"{doc_id} content {index}",
            section="body" if index % 2 == 0 else "head",
            heading_path="H",
            metadata={},
        )
        for index in range(count)
    ]


def _embeds(doc_id: str, count: int, dim: int) -> list[list[float]]:
    position = hash(doc_id) % dim
    return [
        [
            1.0 if j == (position + index) % dim else 0.0
            for j in range(dim)
        ]
        for index in range(count)
    ]


def _payloads(doc_id: str, title: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "title_norm": title.lower(),
        "year": 2024,
        "journal": "TMLR",
        "journal_norm": "tmlr",
        "first_author": "Smith",
        "first_author_norm": "smith",
        "author_count": 1,
        "doi": f"10.1/{doc_id}",
        "source_file": f"/src/{doc_id}.pdf",
        "content_hash": f"hash-{doc_id}",
        "added_at": "2026-08-26T00:00:00Z",
        "added_ts": 1785024000.0,
    }


def _wait_count(store: VectorStore, expected: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if store.count() == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"count did not reach {expected} (got {store.count()})")


def test_create_creates_collection_with_payload_indexes():
    VectorStore.create("alpha", dim=8)
    client = get_qdrant_manager().client()
    info = client.get_collection(collection_name="kb_alpha")
    assert info.config.params.vectors.size == 8
    assert info.config.params.vectors.distance == models.Distance.COSINE
    index_types = {
        field: schema.data_type for field, schema in info.payload_schema.items()
    }
    expected = {
        "doc_id": "keyword",
        "section": "keyword",
        "year": "integer",
        "journal_norm": "keyword",
        "title_norm": "keyword",
        "first_author_norm": "keyword",
        "doi": "keyword",
        "added_ts": "float",
    }
    for field, data_type in expected.items():
        assert index_types[field] == data_type


def test_upsert_query_payloads_two_stage():
    store = VectorStore.create("beta", dim=8)
    chunks = _chunks("doc1", 3)
    embeds = _embeds("doc1", 3, 8)
    ids = store.upsert_chunks("doc1", chunks, embeds, _payloads("doc1", "Doc One"))
    assert len(ids) == 3
    _wait_count(store, 3)

    results = store.query(embeds[0], None, limit=10)
    assert len(results) == 3
    top_id, top_score = results[0]
    assert top_id == ids[0]
    assert abs(top_score - 1.0) < 1e-6

    payloads = store.payloads([ids[0]])
    assert payloads[ids[0]]["doc_id"] == "doc1"
    assert payloads[ids[0]]["title"] == "Doc One"
    assert payloads[ids[0]]["content"] == chunks[0].content
    assert payloads[ids[0]]["chunk_index"] == 0
    assert payloads[ids[0]]["section"] == "body"
    assert payloads[ids[0]]["heading_path"] == "H"


def test_query_with_filter_returns_only_matching_docs():
    store = VectorStore.create("gamma", dim=8)
    store.upsert_chunks("docA", _chunks("docA", 2), _embeds("docA", 2, 8), _payloads("docA", "A"))
    store.upsert_chunks("docB", _chunks("docB", 2), _embeds("docB", 2, 8), _payloads("docB", "B"))
    _wait_count(store, 4)
    query_filter = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="docA"))]
    )
    results = store.query(_embeds("docA", 1, 8)[0], query_filter, limit=10)
    assert len(results) == 2
    payloads = store.payloads([point_id for point_id, _ in results])
    assert all(payloads[point_id]["doc_id"] == "docA" for point_id, _ in results)


def test_reupsert_same_doc_overwrites_and_keeps_count():
    store = VectorStore.create("delta", dim=8)
    store.upsert_chunks("doc1", _chunks("doc1", 3), _embeds("doc1", 3, 8), _payloads("doc1", "V1"))
    _wait_count(store, 3)
    store.upsert_chunks("doc1", _chunks("doc1", 3), _embeds("doc1", 3, 8), _payloads("doc1", "V2"))
    _wait_count(store, 3)
    top = store.query(_embeds("doc1", 1, 8)[0], None, limit=1)[0]
    assert abs(top[1] - 1.0) < 1e-6
    point_id = top[0]
    assert store.payloads([point_id])[point_id]["title"] == "V2"


def test_delete_doc_only_deletes_target():
    store = VectorStore.create("epsilon", dim=8)
    store.upsert_chunks("docA", _chunks("docA", 3), _embeds("docA", 3, 8), _payloads("docA", "A"))
    store.upsert_chunks("docB", _chunks("docB", 2), _embeds("docB", 2, 8), _payloads("docB", "B"))
    _wait_count(store, 5)
    deleted = store.delete_doc("docA")
    assert deleted == 3
    _wait_count(store, 2)
    remaining = store.query(_embeds("docB", 1, 8)[0], None, limit=10)
    assert len(remaining) == 2
    remaining_payloads = store.payloads([point_id for point_id, _ in remaining])
    assert all(remaining_payloads[point_id]["doc_id"] == "docB" for point_id, _ in remaining)


def test_drop_then_open_raises():
    store = VectorStore.create("zeta", dim=8)
    VectorStore.open("zeta")
    store.drop()
    with pytest.raises(ServiceUnavailableError):
        VectorStore.open("zeta")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_auto_spawn_manager_spawns_and_stops(monkeypatch, tmp_path, qdrant_instance):
    port = _free_port()
    storage = tmp_path / "qdrant-storage"
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_URL", "")
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_BIN", str(qdrant_instance["binary"]))
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_PORT", str(port))
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_STORAGE_DIR", str(storage))
    reset_qdrant_manager()
    manager = get_qdrant_manager()
    client = manager.client()
    assert client.get_collections().collections == []
    VectorStore.create("autospawn", dim=8)
    assert client.collection_exists("kb_autospawn")
    manager = get_qdrant_manager()
    process = manager._process
    assert process is not None
    assert process.poll() is None
    reset_qdrant_manager()
    assert process.poll() is not None


def test_auto_download_then_spawn(monkeypatch, tmp_path, qdrant_instance):
    from scholar_rag.store import qdrant_platform

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    archive = release_dir / "qdrant-x86_64-unknown-linux-gnu.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(qdrant_instance["binary"], arcname="qdrant")
    server = HTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(release_dir)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = _free_port()
        storage = tmp_path / "dl-storage"
        monkeypatch.setenv("SCHOLAR_RAG_QDRANT_URL", "")
        monkeypatch.setenv("SCHOLAR_RAG_QDRANT_BIN", "")
        monkeypatch.setenv("SCHOLAR_RAG_QDRANT_PORT", str(port))
        monkeypatch.setenv("SCHOLAR_RAG_QDRANT_STORAGE_DIR", str(storage))
        monkeypatch.setattr(
            qdrant_platform,
            "release_url",
            lambda asset: f"http://127.0.0.1:{server.server_port}/{archive.name}",
        )
        reset_qdrant_manager()
        manager = get_qdrant_manager()
        client = manager.client()
        assert client.get_collections().collections == []
        assert (tmp_path / "data" / "bin" / "qdrant").is_file()
        process = manager._process
        assert process is not None
        assert process.poll() is None
        reset_qdrant_manager()
        assert process.poll() is not None
    finally:
        server.shutdown()
