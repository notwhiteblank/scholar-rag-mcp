"""Real end-to-end smoke test.

Drives the full pipeline with real vLLM model services, real MinerU, and a real
Qdrant binary: create_kb -> get_job -> list/search documents -> search_chunks
(with and without metadata filter) -> get_document -> get_document_text paging ->
remove_document -> delete_kb (two-phase).

Prerequisites (skipped with guidance when missing):
- vLLM chat/embed/rerank models running on 8101/8102/8103 (scripts/serve_models.sh)
- This test must run inside the `mineru` pixi environment:
      pixi run -e mineru pytest tests/e2e/smoke.py -v -m e2e
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import time
from pathlib import Path

import httpx
import pytest

from scholar_rag.server.tools import dispatch_tool

pytestmark = pytest.mark.e2e

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_VLLM_PORTS = {"chat": 8101, "embed": 8102, "rerank": 8103}
_JOB_TIMEOUT = 1800.0


def _env_ready() -> tuple[bool, str]:
    try:
        importlib.util.find_spec("mineru.cli.common")
    except (ImportError, ModuleNotFoundError):
        return False, (
            "mineru is not importable in this environment; run the smoke test in the "
            "mineru pixi environment: pixi run -e mineru pytest tests/e2e/smoke.py -m e2e"
        )
    for role, port in _VLLM_PORTS.items():
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            if response.status_code != 200:
                raise httpx.HTTPError(f"HTTP {response.status_code}")
        except httpx.HTTPError as exc:
            return False, (
                f"vllm {role} model not ready on :{port} ({exc}); start the three models "
                "with scripts/serve_models.sh before running the smoke test"
            )
    return True, ""


def served_model_ids() -> dict[str, str]:
    ids: dict[str, str] = {}
    for role, port in _VLLM_PORTS.items():
        payload = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5).json()
        ids[role] = payload["data"][0]["id"]
    return ids


def _call(name: str, arguments: dict) -> dict:
    result = dispatch_tool(name, arguments)
    assert "error_code" not in result, (
        f"tool {name} returned error: {result.get('error_code')} "
        f"message={result.get('message')} hint={result.get('hint')}"
    )
    return result


def _wait_job(kb: str, job_id: str) -> dict:
    deadline = time.monotonic() + _JOB_TIMEOUT
    while time.monotonic() < deadline:
        record = _call("get_job", {"job_id": job_id})
        if record["status"] in ("succeeded", "failed"):
            return record
        time.sleep(1.0)
    raise AssertionError(
        f"job {job_id} for kb {kb} did not complete within {_JOB_TIMEOUT}s; "
        f"last status: {record.get('status')}"
    )


def _barf(record: dict) -> str:
    return (
        f"record: {record}"
        if not record.get("result_summary")
        else f"result_summary: {record['result_summary']}"
    )


def test_full_chain_smoke(monkeypatch, tmp_path) -> None:
    ready, guidance = _env_ready()
    if not ready:
        pytest.skip(guidance)

    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    data_dir = tmp_path / "data"
    storage = tmp_path / "qdrant"
    spike_binary = Path(__file__).resolve().parents[2] / "spike" / "out" / "qdrant" / "qdrant"
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_STORAGE_DIR", str(storage))
    if spike_binary.is_file():
        monkeypatch.setenv("SCHOLAR_RAG_QDRANT_BIN", str(spike_binary))
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_BASE_URL", "http://127.0.0.1:8101/v1")
    monkeypatch.setenv("SCHOLAR_RAG_EMBED_BASE_URL", "http://127.0.0.1:8102/v1")
    monkeypatch.setenv("SCHOLAR_RAG_RERANK_BASE_URL", "http://127.0.0.1:8103/v1")
    for role, model_id in served_model_ids().items():
        monkeypatch.setenv(f"SCHOLAR_RAG_{role.upper()}_MODEL", model_id)

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    for name in ("paper_a", "paper_b", "paper_c"):
        shutil.copy2(_FIXTURES / f"{name}.pdf", papers_dir / f"{name}.pdf")

    created = _call("create_kb", {"kb_name": "smoke", "folder_path": str(papers_dir)})
    record = _wait_job("smoke", created["job_id"])
    assert record["status"] == "succeeded", f"create_kb failed: {_barf(record)}"
    assert record["progress"] == {"done": 3, "total": 3}
    assert record["result_summary"]["succeeded"] == 3, _barf(record)
    assert record["result_summary"]["failed"] == 0, _barf(record)
    assert "elapsed_s" in record["timings"]

    listed = _call("list_documents", {"kb": "smoke", "page_size": 20})
    assert listed["total"] == 3
    doc_ids = [row["doc_id"] for row in listed["results"]]
    assert len(doc_ids) == 3
    hit = _call("search_documents", {"kb": "smoke", "title": "Wireless Network Routing"})
    assert hit["total"] == 1, f"expected one wireless routing paper, got {hit}"
    assert hit["results"][0]["title"].startswith("Graph Neural Networks")

    chunks = _call("search_chunks", {"kb": "smoke", "query": "yield forecasting remote sensing", "top_k": 5})
    assert chunks["chunks"], "search_chunks returned no results"
    first = chunks["chunks"][0]
    assert first["embed_score"] is not None
    assert first["rerank_score"] is not None, "reranker did not score candidates"

    filtered = _call(
        "search_chunks",
        {"kb": "smoke", "query": "yield forecasting", "top_k": 10,
         "metadata_filter": {"doc_id": [doc_ids[0]]}},
    )
    assert filtered["chunks"], "filtered search returned no results"
    for chunk in filtered["chunks"]:
        assert chunk["metadata"]["doc_id"] == doc_ids[0], (
            f"filter leaked doc {chunk['metadata']['doc_id']} outside {doc_ids[0]}"
        )

    overview = _call("get_document", {"kb": "smoke", "doc_id": doc_ids[0]})
    assert overview["outline"], "get_document returned an empty outline"
    assert overview["total_chars"] > 0

    target = doc_ids[0]
    page_size = max(1, overview["total_chars"] // 3)
    total_pages = None
    reassembled = ""
    expected_chars = None
    page_number = 1
    while True:
        page = _call(
            "get_document_text",
            {"kb": "smoke", "doc_id": target, "page": page_number, "page_size": page_size},
        )
        total_pages = page["total_pages"]
        expected_chars = page["total_chars"]
        reassembled += page["content"]
        if page["next_hint"] is None:
            break
        page_number += 1
    assert page_number == total_pages, f"walked {page_number} pages but total_pages={total_pages}"
    assert len(reassembled) == expected_chars, (
        f"reassembled {len(reassembled)} chars, expected {expected_chars}"
    )
    out_of_range = dispatch_tool(
        "get_document_text",
        {"kb": "smoke", "doc_id": target, "page": total_pages + 1, "page_size": page_size},
    )
    assert out_of_range["error_code"] == "invalid_request"
    assert f"pages 1..{total_pages}" in out_of_range["message"]
    missing_section = dispatch_tool(
        "get_document_text",
        {"kb": "smoke", "doc_id": target, "section": "NoSuchSection"},
    )
    assert missing_section["error_code"] == "invalid_request"

    removed = _call("remove_document", {"kb": "smoke", "doc_id": target})
    assert removed["chunks_deleted"] >= 1
    assert removed["catalog_deleted"] is True
    assert removed["files_deleted"] is True
    after_remove = _call("list_documents", {"kb": "smoke", "page_size": 20})
    assert after_remove["total"] == 2
    gone = _call(
        "search_chunks",
        {"kb": "smoke", "query": "yield forecasting", "top_k": 10,
         "metadata_filter": {"doc_id": [target]}},
    )
    assert gone["chunks"] == [], "removed document still matches search_chunks"

    preview = _call("delete_kb", {"kb": "smoke"})
    assert preview["doc_count"] == 2
    assert preview["chunk_count"] >= 2
    assert "deleted" not in preview
    confirmed = _call("delete_kb", {"kb": "smoke", "confirm_token": preview["confirm_token"]})
    assert confirmed == {"kb": "smoke", "deleted": True}
    kbs = _call("list_kbs", {})
    assert all(kb["name"] != "smoke" for kb in kbs["kbs"])
    assert not (data_dir / "kbs" / "smoke").exists()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "-m", "e2e"]))
