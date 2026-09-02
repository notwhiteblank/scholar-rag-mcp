from __future__ import annotations

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

DEFAULT_MODEL_NAME = "jina-embeddings-v5-text-small"


class _ApiEmbeddings(OpenAICompatClient):
    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._post("/embeddings", {"model": self.model, "input": batch})
            if response.status_code >= 400:
                raise ServiceUnavailableError(
                    f"embedding failed: HTTP {response.status_code}"
                )
            expected = len(batch)
            items = response.json().get("data", [])
            if len(items) != expected:
                raise ServiceUnavailableError(
                    "embedding response length mismatch: "
                    f"expected {expected} vectors, got {len(items)}"
                )
            try:
                indices = [int(item.get("index", -1)) for item in items]
            except (KeyError, TypeError, ValueError):
                raise ServiceUnavailableError(
                    "embedding response contains an item with an invalid index"
                ) from None
            if sorted(indices) != list(range(expected)):
                raise ServiceUnavailableError(
                    "embedding response index mismatch: expected indices "
                    f"0..{expected - 1} exactly once, got indices {indices}"
                )
            ordered = sorted(items, key=lambda item: int(item.get("index", -1)))
            for item in ordered:
                all_embeddings.append(list(item["embedding"]))
        return all_embeddings


class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = DEFAULT_MODEL_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        self._api = _ApiEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return self._api.embed(texts, batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def health(self) -> bool:
        return self._api.health()

    def close(self) -> None:
        self._api.close()
