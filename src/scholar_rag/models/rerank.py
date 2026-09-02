from __future__ import annotations

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

DEFAULT_MODEL_NAME = "jina-reranker-v3.5"


class _ApiReranker(OpenAICompatClient):
    def rerank(self, query: str, documents: list[str], batch_size: int = 128) -> list[float]:
        if not documents:
            return []
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = self._post(
                "/rerank",
                {"model": self.model, "query": query, "documents": batch, "top_n": 0},
            )
            if response.status_code >= 400:
                raise ServiceUnavailableError(f"rerank failed: HTTP {response.status_code}")
            expected = len(batch)
            results = response.json().get("results", [])
            try:
                result_indices = [int(result["index"]) for result in results]
            except (KeyError, TypeError, ValueError):
                raise ServiceUnavailableError(
                    "rerank response contains a result with a missing or invalid index"
                ) from None
            if sorted(result_indices) != list(range(expected)):
                raise ServiceUnavailableError(
                    "rerank response index mismatch: expected "
                    f"{expected} scores covering indices 0..{expected - 1} exactly once, "
                    f"got {len(results)} results with indices {result_indices}"
                )
            chunk_scores = [0.0] * expected
            for result in results:
                chunk_scores[int(result["index"])] = float(result["relevance_score"])
            scores.extend(chunk_scores)
        return scores


class RerankClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = DEFAULT_MODEL_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        self._api = _ApiReranker(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    def rerank(self, query: str, documents: list[str], batch_size: int = 128) -> list[float]:
        return self._api.rerank(query, documents, batch_size)

    def health(self) -> bool:
        return self._api.health()

    def close(self) -> None:
        self._api.close()
