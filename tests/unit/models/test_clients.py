import os

import pytest

import scholar_rag.models as models
from scholar_rag.models import (
    get_chat_client,
    get_embedding_client,
    get_rerank_client,
    reset_clients,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_BASE_URL", "http://127.0.0.1:8101/v1")
    monkeypatch.setenv("SCHOLAR_RAG_EMBED_BASE_URL", "http://127.0.0.1:8102/v1")
    monkeypatch.setenv("SCHOLAR_RAG_RERANK_BASE_URL", "http://127.0.0.1:8103/v1")
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_API_KEY", "k")
    monkeypatch.setenv("SCHOLAR_RAG_EMBED_API_KEY", "k")
    monkeypatch.setenv("SCHOLAR_RAG_RERANK_API_KEY", "k")
    reset_clients()


def test_no_clients_constructed_at_import():
    assert models._chat_client is None
    assert models._embedding_client is None
    assert models._rerank_client is None


def test_get_chat_client_returns_singleton():
    assert get_chat_client() is get_chat_client()


def test_get_embedding_client_returns_singleton():
    assert get_embedding_client() is get_embedding_client()


def test_get_rerank_client_returns_singleton():
    assert get_rerank_client() is get_rerank_client()


def test_reset_clients_clears_singletons():
    first = get_embedding_client()
    reset_clients()
    assert models._embedding_client is None
    assert get_embedding_client() is not first


def test_singletons_are_distinct_clients():
    chat = get_chat_client()
    embed = get_embedding_client()
    rerank = get_rerank_client()
    assert chat is not embed
    assert embed is not rerank
    assert chat is not rerank
