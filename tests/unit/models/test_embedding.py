import json
import sys
import time
import types

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


FAKE_EMBED_SCRIPT = """
class _Tensor:
    def __init__(self, rows):
        self._rows = rows

    def cpu(self):
        return self

    def float(self):
        return self

    def tolist(self):
        return self._rows


class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path=None, **kwargs):
        self._seen = 0

    def process(self, inputs, normalize=True):
        rows = []
        for item in inputs:
            rows.append([float(self._seen), 5.0])
            self._seen += 1
        return _Tensor(rows)
"""


@pytest.fixture
def fake_local_model(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "qwen3_vl_embedding.py").write_text(FAKE_EMBED_SCRIPT)
    return tmp_path


def test_local_embed_order_and_batch_keep(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16=1))
    client = EmbeddingClient(backend="local", model=str(fake_local_model))
    result = client.embed([f"s{i}" for i in range(10)], batch_size=3)
    assert len(result) == 10
    assert [row[0] for row in result] == [float(i) for i in range(10)]
    assert all(row[1] == 5.0 for row in result)


def test_local_embed_empty_returns_empty(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(float16=1))
    client = EmbeddingClient(backend="local", model=str(fake_local_model))
    assert client.embed([]) == []


def test_local_missing_script_raises_service_unavailable(tmp_path):
    client = EmbeddingClient(backend="local", model=str(tmp_path))
    with pytest.raises(ServiceUnavailableError):
        client.embed(["t0"])


def test_api_client_never_imports_torch(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "torch", None)
    with respx.mock:
        respx.post(EMBED_URL).mock(side_effect=_embedding_payload)
        client.embed(["t0"])
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={}))
        assert client.health() is True


def test_local_construction_does_not_import_torch(monkeypatch, fake_local_model):
    monkeypatch.setitem(sys.modules, "torch", None)
    EmbeddingClient(backend="local", model=str(fake_local_model))
