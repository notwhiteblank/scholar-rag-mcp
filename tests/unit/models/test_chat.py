import json
import sys
import time
import types

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


class _FakeInputIds:
    shape = (1, 3)


class _FakeBatch(dict):
    def __init__(self):
        super().__init__()
        self["input_ids"] = _FakeInputIds()

    def to(self, device):
        return self


class _FakeSeq:
    def __getitem__(self, key):
        return [9, 10, 11]


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _FakeModel:
    device = "cpu"
    generate_kwargs = None
    generate_calls = 0

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def eval(self):
        pass

    def generate(self, **kwargs):
        type(self).generate_kwargs = kwargs
        type(self).generate_calls += 1
        return _FakeSeq()


class _FakeTokenizer:
    response = ""
    decode_calls = 0
    applied_messages = None
    enable_thinking_kwarg = "UNSET"
    reject_thinking_kwarg = False

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        if type(self).reject_thinking_kwarg and "enable_thinking" in kwargs:
            raise TypeError("unknown kwarg")
        type(self).applied_messages = messages
        type(self).enable_thinking_kwarg = kwargs.get("enable_thinking", "UNSET")
        return "<template>"

    def __call__(self, text, return_tensors=None):
        return _FakeBatch()

    def decode(self, tokens, skip_special_tokens=True):
        type(self).decode_calls += 1
        return type(self).response


@pytest.fixture
def fake_local_chat_deps(monkeypatch):
    fake_torch = types.SimpleNamespace(
        bfloat16=1,
        no_grad=lambda: _NoGrad(),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        ),
    )
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=_FakeModel,
        AutoTokenizer=_FakeTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _FakeTokenizer.response = ""
    _FakeTokenizer.decode_calls = 0
    _FakeTokenizer.applied_messages = None
    _FakeTokenizer.enable_thinking_kwarg = "UNSET"
    _FakeTokenizer.reject_thinking_kwarg = False
    _FakeModel.generate_kwargs = None
    _FakeModel.generate_calls = 0


def test_local_chat_load_is_deferred(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    ChatClient(backend="local", model="/models/fake-chat")


def test_local_chat_complete_returns_text(fake_local_chat_deps):
    _FakeTokenizer.response = "hello world"
    client = ChatClient(backend="local", model="/models/fake-chat")
    assert client.complete("hi") == "hello world"
    assert _FakeTokenizer.applied_messages == [{"role": "user", "content": "hi"}]
    assert _FakeTokenizer.enable_thinking_kwarg is False
    assert _FakeModel.generate_kwargs["temperature"] == 0.1
    assert _FakeModel.generate_kwargs["top_p"] == 0.9
    assert _FakeModel.generate_kwargs["do_sample"] is True


def test_local_chat_json_schema_parses_fenced_json(fake_local_chat_deps):
    _FakeTokenizer.response = '```json\n{"ok": true}\n```'
    client = ChatClient(backend="local", model="/models/fake-chat")
    result = client.complete("hi", json_schema={"type": "object"})
    assert result == {"ok": True}
    content = _FakeTokenizer.applied_messages[0]["content"]
    assert "type" in content
    assert _FakeTokenizer.decode_calls == 1


def test_local_chat_falls_back_without_thinking_kwarg(fake_local_chat_deps):
    _FakeTokenizer.reject_thinking_kwarg = True
    _FakeTokenizer.response = '{"ok": 1}'
    client = ChatClient(backend="local", model="/models/fake-chat")
    assert client.complete("hi", json_schema={"type": "object"}) == {"ok": 1}
    assert _FakeTokenizer.enable_thinking_kwarg == "UNSET"


def test_local_chat_invalid_json_retries_then_raises(fake_local_chat_deps):
    _FakeTokenizer.response = "not json"
    client = ChatClient(backend="local", model="/models/fake-chat")
    with pytest.raises(ServiceUnavailableError) as exc_info:
        client.complete("hi", json_schema={"type": "object"})
    assert "invalid json after retries" in str(exc_info.value)
    assert _FakeTokenizer.decode_calls == 3
