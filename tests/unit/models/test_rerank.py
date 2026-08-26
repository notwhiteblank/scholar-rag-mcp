import json
import sys
import time
import types

import httpx
import pytest
import respx

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.rerank import RerankClient

BASE_URL = "http://127.0.0.1:8103/v1"
RERANK_URL = "http://127.0.0.1:8103/v1/rerank"
MODELS_URL = "http://127.0.0.1:8103/v1/models"


@pytest.fixture
def client() -> RerankClient:
    return RerankClient(base_url=BASE_URL, api_key="k", model="m")


def _rerank_payload(request) -> httpx.Response:
    body = json.loads(request.content)
    documents = body["documents"]
    results = []
    for i, doc in enumerate(documents):
        global_index = int(doc.removeprefix("d"))
        results.append(
            {
                "index": i,
                "document": {"text": doc},
                "relevance_score": float(global_index) / 100.0,
            }
        )
    return httpx.Response(
        200,
        json={
            "id": "score-1",
            "model": body["model"],
            "usage": {"prompt_tokens": 31, "total_tokens": 31},
            "results": results,
        },
    )


@respx.mock
def test_rerank_single_batch_scores_in_original_order(client):
    route = respx.post(RERANK_URL).mock(side_effect=_rerank_payload)
    result = client.rerank("query", ["d2", "d0", "d1"])
    assert result == [0.02, 0.0, 0.01]
    assert route.call_count == 1


@respx.mock
def test_rerank_out_of_order_results_mapped_back(client):
    def handler(request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "document": {"text": "c"}, "relevance_score": 0.9},
                    {"index": 0, "document": {"text": "a"}, "relevance_score": 0.1},
                    {"index": 1, "document": {"text": "b"}, "relevance_score": 0.5},
                ]
            },
        )

    route = respx.post(RERANK_URL).mock(side_effect=handler)
    assert client.rerank("query", ["a", "b", "c"]) == [0.1, 0.5, 0.9]
    assert route.call_count == 1


@respx.mock
def test_rerank_batches_keep_original_order(client):
    documents = [f"d{i}" for i in range(200)]
    route = respx.post(RERANK_URL).mock(side_effect=_rerank_payload)
    result = client.rerank("query", documents, batch_size=128)
    assert len(result) == 200
    assert result == [float(i) / 100.0 for i in range(200)]
    assert route.call_count == 2


@respx.mock
def test_rerank_request_schema(client):
    def plain_handler(request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": i, "document": {"text": doc}, "relevance_score": float(i) / 10.0}
                    for i, doc in enumerate(body["documents"])
                ]
            },
        )

    route = respx.post(RERANK_URL).mock(side_effect=plain_handler)
    client.rerank("my query", ["a", "b"])
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "model": "m",
        "query": "my query",
        "documents": ["a", "b"],
        "top_n": 0,
    }


@respx.mock
def test_5xx_retried_then_succeeds(client, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(RERANK_URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(
                200,
                json={"results": [{"index": 0, "relevance_score": 0.4}]},
            ),
        ]
    )
    assert client.rerank("q", ["a"]) == [0.4]
    assert route.call_count == 2


@respx.mock
def test_4xx_not_retried_and_raises(client):
    route = respx.post(RERANK_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad"}})
    )
    with pytest.raises(ServiceUnavailableError):
        client.rerank("q", ["a"])
    assert route.call_count == 1


@respx.mock
def test_health_true(client):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    assert client.health() is True


def test_rerank_empty_returns_empty(client):
    assert client.rerank("q", []) == []


FAKE_RERANK_SCRIPT = """
class Qwen3VLReranker:
    def __init__(self, model_name_or_path=None, **kwargs):
        self._seen = 0

    def process(self, inputs):
        scores = []
        for _doc in inputs["documents"]:
            if self._seen % 2 == 0:
                scores.append(0.8)
            else:
                scores.append(0.2)
            self._seen += 1
        return scores
"""


@pytest.fixture
def fake_local_model(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "qwen3_vl_reranker.py").write_text(FAKE_RERANK_SCRIPT)
    return tmp_path


def test_local_rerank_order_kept(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16=1))
    client = RerankClient(backend="local", model=str(fake_local_model))
    result = client.rerank("q", [f"d{i}" for i in range(5)], batch_size=2)
    assert result == [0.8, 0.2, 0.8, 0.2, 0.8]


def test_local_rerank_empty_returns_empty(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16=1))
    client = RerankClient(backend="local", model=str(fake_local_model))
    assert client.rerank("q", []) == []


def test_local_missing_script_raises_service_unavailable(tmp_path):
    client = RerankClient(backend="local", model=str(tmp_path))
    with pytest.raises(ServiceUnavailableError):
        client.rerank("q", ["d0"])


def test_api_client_never_imports_torch(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "torch", None)
    with respx.mock:
        respx.post(RERANK_URL).mock(side_effect=_rerank_payload)
        client.rerank("q", ["d0"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={}))
        assert client.health() is True


def test_local_construction_does_not_import_torch(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", None)
    RerankClient(backend="local", model=str(fake_local_model))
