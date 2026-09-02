from __future__ import annotations

from threading import Lock

from scholar_rag.core.config import Settings
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT
from scholar_rag.models.chat import ChatClient
from scholar_rag.models.embedding import EmbeddingClient
from scholar_rag.models.rerank import RerankClient

_clients_lock = Lock()
_chat_client: ChatClient | None = None
_embedding_client: EmbeddingClient | None = None
_rerank_client: RerankClient | None = None


def get_chat_client() -> ChatClient:
    global _chat_client
    with _clients_lock:
        if _chat_client is None:
            settings = Settings.load()
            _chat_client = ChatClient(
                base_url=settings.chat_base_url,
                api_key=settings.chat_api_key,
                model=settings.chat_model,
                timeout=DEFAULT_TIMEOUT,
                max_retries=DEFAULT_HTTP_MAX_RETRIES,
            )
        return _chat_client


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    with _clients_lock:
        if _embedding_client is None:
            settings = Settings.load()
            _embedding_client = EmbeddingClient(
                base_url=settings.embed_base_url,
                api_key=settings.embed_api_key,
                model=settings.embed_model,
                timeout=DEFAULT_TIMEOUT,
                max_retries=DEFAULT_HTTP_MAX_RETRIES,
            )
        return _embedding_client


def get_rerank_client() -> RerankClient:
    global _rerank_client
    with _clients_lock:
        if _rerank_client is None:
            settings = Settings.load()
            _rerank_client = RerankClient(
                base_url=settings.rerank_base_url,
                api_key=settings.rerank_api_key,
                model=settings.rerank_model,
                timeout=DEFAULT_TIMEOUT,
                max_retries=DEFAULT_HTTP_MAX_RETRIES,
            )
        return _rerank_client


def reset_clients() -> None:
    global _chat_client, _embedding_client, _rerank_client
    with _clients_lock:
        for client in (_chat_client, _embedding_client, _rerank_client):
            if client is not None:
                client.close()
        _chat_client = None
        _embedding_client = None
        _rerank_client = None


__all__ = [
    "ChatClient",
    "EmbeddingClient",
    "RerankClient",
    "get_chat_client",
    "get_embedding_client",
    "get_rerank_client",
    "reset_clients",
]
