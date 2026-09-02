import json
import sys
import time

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


@respx.mock
def test_rerank_duplicate_index_raises(client):
    route = respx.post(RERANK_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 0, "relevance_score": 0.7},
                ]
            },
        )
    )
    with pytest.raises(ServiceUnavailableError) as exc_info:
        client.rerank("q", ["a", "b"])
    message = str(exc_info.value)
    assert "expected 2" in message
    assert "exactly once" in message
    assert route.call_count == 1


@respx.mock
def test_rerank_missing_index_raises(client):
    route = respx.post(RERANK_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"index": 0, "relevance_score": 0.5}]},
        )
    )
    with pytest.raises(ServiceUnavailableError):
        client.rerank("q", ["a", "b"])
    assert route.call_count == 1


@respx.mock
def test_rerank_out_of_range_index_raises(client):
    route = respx.post(RERANK_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"index": 3, "relevance_score": 0.5}]},
        )
    )
    with pytest.raises(ServiceUnavailableError):
        client.rerank("q", ["a", "b"])
    assert route.call_count == 1


def test_rerank_empty_returns_empty(client):
    assert client.rerank("q", []) == []


def test_api_client_never_imports_torch(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "torch", None)
    with respx.mock:
        respx.post(RERANK_URL).mock(side_effect=_rerank_payload)
        client.rerank("q", ["d0"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={}))
        assert client.health() is True


def test_construction_with_defaults():
    client = RerankClient(base_url=BASE_URL, api_key="k")
    assert client.rerank("q", []) == []
