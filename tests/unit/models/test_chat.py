import json
import time

import httpx
import pytest
import respx

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.chat import ChatClient

BASE_URL = "http://127.0.0.1:8101/v1"
CHAT_URL = "http://127.0.0.1:8101/v1/chat/completions"


@pytest.fixture
def client() -> ChatClient:
    return ChatClient(base_url=BASE_URL, api_key="k", model="m")


def _body(request) -> dict:
    return json.loads(request.content)


def _completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@respx.mock
def test_complete_returns_string_text(client):
    route = respx.post(CHAT_URL).mock(return_value=_completion("hello world"))
    result = client.complete("hi")
    assert result == "hello world"
    assert route.call_count == 1


@respx.mock
def test_complete_with_json_schema_returns_dict(client):
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    route = respx.post(CHAT_URL).mock(return_value=_completion('{"a": 1}'))
    result = client.complete("hi", json_schema=schema)
    assert result == {"a": 1}
    assert route.call_count == 1
    payload = _body(route.calls[0].request)
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "schema", "schema": schema, "strict": True},
    }


@respx.mock
def test_json_schema_4xx_falls_back_to_prompt_constraint(client):
    schema = {"type": "object"}
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unsupported response_format"}}),
            _completion('{"ok": true}'),
        ]
    )
    result = client.complete("hi", json_schema=schema)
    assert result == {"ok": True}
    assert route.call_count == 2
    fallback_body = _body(route.calls[1].request)
    assert "response_format" not in fallback_body
    assert schema["type"] in fallback_body["messages"][0]["content"]


@respx.mock
def test_json_schema_invalid_json_falls_back_to_prompt_constraint(client):
    schema = {"type": "object"}
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            _completion("not-json"),
            _completion('{"ok": true}'),
        ]
    )
    result = client.complete("hi", json_schema=schema)
    assert result == {"ok": True}
    assert route.call_count == 2


@respx.mock
def test_invalid_json_after_retries_raises_service_unavailable(client):
    schema = {"type": "object"}
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unsupported"}}),
            _completion("garbage-1"),
            _completion("garbage-2"),
            _completion("garbage-3"),
        ]
    )
    with pytest.raises(ServiceUnavailableError) as exc_info:
        client.complete("hi", json_schema=schema)
    assert "invalid json after retries" in str(exc_info.value)
    assert route.call_count == 4


@respx.mock
def test_5xx_retried_then_succeeds_on_json_schema_path(client, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(502, json={"error": "bad gateway"}),
            _completion('{"a": 2}'),
        ]
    )
    result = client.complete("hi", json_schema={"type": "object"})
    assert result == {"a": 2}
    assert route.call_count == 2


@respx.mock
def test_complete_failure_raises_service_unavailable(client):
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(400, json={}))
    with pytest.raises(ServiceUnavailableError):
        client.complete("hi")
    assert route.call_count == 1


@respx.mock
def test_5xx_exhausted_on_plain_path_raises(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    route = respx.post(CHAT_URL).mock(return_value=httpx.Response(503, json={}))
    client = ChatClient(base_url=BASE_URL, api_key="k", model="m", max_retries=2)
    with pytest.raises(ServiceUnavailableError):
        client.complete("hi")
    assert route.call_count == 3


@respx.mock
def test_prompt_constraint_path_4xx_raises_service_unavailable(client):
    route = respx.post(CHAT_URL).mock(
        side_effect=[
            httpx.Response(400, json={}),
            httpx.Response(401, json={"error": "unauthorized"}),
        ]
    )
    with pytest.raises(ServiceUnavailableError):
        client.complete("hi", json_schema={"type": "object"})
    assert route.call_count == 2


@respx.mock
def test_construction_with_defaults():
    route = respx.post(CHAT_URL).mock(return_value=_completion("hello"))
    client = ChatClient(base_url=BASE_URL, api_key="k")
    assert client.complete("hi") == "hello"
    assert route.call_count == 1
