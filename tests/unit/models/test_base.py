import time

import httpx
import pytest
import respx

from scholar_rag.core.errors import ConfigError, ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_TIMEOUT, OpenAICompatClient

BASE_URL = "http://127.0.0.1:8101/v1"
CHAT_URL = "http://127.0.0.1:8101/v1/chat/completions"
MODELS_URL = "http://127.0.0.1:8101/v1/models"


@pytest.fixture
def client() -> OpenAICompatClient:
    return OpenAICompatClient(base_url=BASE_URL, api_key="k", model="m")


def test_default_timeout_is_120_seconds(client):
    assert client.timeout == 120.0
    assert DEFAULT_TIMEOUT == 120.0


def test_base_url_trailing_slash_normalized():
    c = OpenAICompatClient(base_url="http://127.0.0.1:8101/v1/", api_key="k", model="m")
    assert c.base_url == BASE_URL


def test_empty_base_url_raises_config_error():
    with pytest.raises(ConfigError):
        OpenAICompatClient(base_url="", api_key="k", model="m")


@respx.mock
def test_4xx_is_not_retried(client):
    route = respx.post(CHAT_URL).mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    response = client._post("/chat/completions", {"model": "m"})
    assert response.status_code == 400
    assert route.call_count == 1


@respx.mock
def test_5xx_retries_then_succeeds(client, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": "boom"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    response = client._post("/chat/completions", {"model": "m"})
    assert response.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_5xx_exhausted_raises_service_unavailable(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    c = OpenAICompatClient(base_url=BASE_URL, api_key="k", model="m", max_retries=2)
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(503, json={"error": "db"}))
    with pytest.raises(ServiceUnavailableError):
        c._post("/chat/completions", {"model": "m"})
    assert route.call_count == 3


@respx.mock
def test_connection_error_retries_then_succeeds(client, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.ConnectError("connection refused"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    response = client._post("/chat/completions", {"model": "m"})
    assert response.status_code == 200
    assert route.call_count == 2


@respx.mock
def test_connection_error_exhausted_raises_service_unavailable(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    c = OpenAICompatClient(base_url=BASE_URL, api_key="k", model="m", max_retries=1)
    route = respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    with pytest.raises(ServiceUnavailableError):
        c._post("/chat/completions", {"model": "m"})
    assert route.call_count == 2


@respx.mock
def test_health_true_on_2xx(client):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))
    assert client.health() is True


@respx.mock
def test_health_false_on_5xx(client):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(500))
    assert client.health() is False


@respx.mock
def test_health_false_on_connection_error(client):
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("offline"))
    assert client.health() is False
