import json
import sys
import time

import httpx
import pytest
import respx

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.embedding import EmbeddingClient

BASE_URL = "http://127.0.0.1:8102/v1"
EMBED_URL = "http://127.0.0.1:8102/v1/embeddings"
MODELS_URL = "http://127.0.0.1:8102/v1/models"


@pytest.fixture
def client() -> EmbeddingClient:
    return EmbeddingClient(base_url=BASE_URL, api_key="k", model="m")


def _embedding_payload(request) -> httpx.Response:
    body = json.loads(request.content)
    texts: list[str] = body["input"]
    data = []
    for i, text in enumerate(texts):
        global_index = int(text.removeprefix("t"))
        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": [float(global_index), 1.0, 0.0],
            }
        )
    return httpx.Response(200, json={"object": "list", "data": data})


@respx.mock
def test_embed_single_batch_in_order(client):
    route = respx.post(EMBED_URL).mock(side_effect=_embedding_payload)
    result = client.embed(["t0", "t1", "t2"])
    assert result == [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    assert route.call_count == 1


@respx.mock
def test_embed_batches_keep_original_order(client):
    texts = [f"t{i}" for i in range(200)]
    route = respx.post(EMBED_URL).mock(side_effect=_embedding_payload)
    result = client.embed(texts, batch_size=64)
    assert len(result) == 200
    assert [row[0] for row in result] == [float(i) for i in range(200)]
    assert route.call_count == 4


def test_embed_empty_returns_empty(client):
    assert client.embed([]) == []


@respx.mock
def test_embed_query_returns_single_vector(client):
    route = respx.post(EMBED_URL).mock(side_effect=_embedding_payload)
    assert client.embed_query("t7") == [7.0, 1.0, 0.0]
    assert route.call_count == 1


@respx.mock
def test_5xx_retried_then_succeeds(client, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(EMBED_URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(200, json={"object": "list", "data": [{"index": 0, "embedding": [1.0]}]}),
        ]
    )
    assert client.embed(["t0"]) == [[1.0]]
    assert route.call_count == 2


@respx.mock
def test_4xx_not_retried_and_raises(client):
    route = respx.post(EMBED_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad"}})
    )
    with pytest.raises(ServiceUnavailableError):
        client.embed(["t0"])
    assert route.call_count == 1


@respx.mock
def test_health_true(client):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    assert client.health() is True


@respx.mock
def test_embed_fewer_vectors_raises(client):
    route = respx.post(EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": [{"index": 0, "embedding": [1.0]}]},
        )
    )
    with pytest.raises(ServiceUnavailableError) as exc_info:
        client.embed(["t0", "t1"])
    message = str(exc_info.value)
    assert "expected 2" in message
    assert "got 1" in message
    assert route.call_count == 1


@respx.mock
def test_embed_more_vectors_raises(client):
    route = respx.post(EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [2.0]},
                    {"index": 2, "embedding": [3.0]},
                ],
            },
        )
    )
    with pytest.raises(ServiceUnavailableError):
        client.embed(["t0", "t1"])
    assert route.call_count == 1


@respx.mock
def test_embed_duplicate_index_raises(client):
    route = respx.post(EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 0, "embedding": [2.0]},
                ],
            },
        )
    )
    with pytest.raises(ServiceUnavailableError):
        client.embed(["t0", "t1"])
    assert route.call_count == 1


@respx.mock
def test_embed_out_of_range_index_raises(client):
    route = respx.post(EMBED_URL).mock(
        return_value=httpx.Response(
            200,
            json={"object": "list", "data": [{"index": 5, "embedding": [1.0]}]},
        )
    )
    with pytest.raises(ServiceUnavailableError):
        client.embed(["t0", "t1"])
    assert route.call_count == 1


def test_api_client_never_imports_torch(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "torch", None)
    with respx.mock:
        respx.post(EMBED_URL).mock(side_effect=_embedding_payload)
        client.embed(["t0"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={}))
        assert client.health() is True


def test_construction_with_defaults():
    client = EmbeddingClient(base_url=BASE_URL, api_key="k")
    assert client.embed([]) == []
