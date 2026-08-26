from __future__ import annotations

import time
from dataclasses import dataclass

from scholar_rag.core.types import RetrievedChunk
from scholar_rag.models import get_embedding_client, get_rerank_client
from scholar_rag.retrieve.filters import translate_filter
from scholar_rag.store.vector_store import VectorStore


@dataclass
class SearchOutcome:
    chunks: list[RetrievedChunk]
    timings: dict[str, float]
    warning: str | None


class RetrievalEngine:
    def __init__(self, kb: str) -> None:
        self._store = VectorStore.open(kb)
        self._embeddings = get_embedding_client()
        self._reranker = get_rerank_client()

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float | None = None,
        metadata_filter: dict | None = None,  # type: ignore[type-arg]
    ) -> SearchOutcome:
        if not 1 <= top_k <= 50:
            raise ValueError(f"top_k must be in 1..50, got {top_k}")
        if min_score is not None and not 0 <= min_score <= 1:
            raise ValueError(f"min_score must be in 0..1, got {min_score}")

        embed_start = time.perf_counter()
        vector = self._embeddings.embed_query(query)
        embed_ms = self._millis(embed_start)

        qfilter = None
        if metadata_filter:
            qfilter = translate_filter(metadata_filter)

        query_start = time.perf_counter()
        candidates = self._store.query(vector, qfilter, limit=min(top_k * 5, 200))
        search_ms = self._millis(query_start)

        if not candidates:
            return SearchOutcome(
                chunks=[],
                timings={"embed_ms": embed_ms, "search_ms": search_ms, "rerank_ms": 0.0},
                warning=None,
            )

        payloads = self._store.payloads([point_id for point_id, _score in candidates])
        rerank_start = time.perf_counter()
        try:
            scores = self._reranker.rerank(
                query,
                [payloads.get(point_id, {}).get("content", "") for point_id, _score in candidates],
            )
        except Exception:
            rerank_ms = self._millis(rerank_start)
            return SearchOutcome(
                chunks=self._build_degraded(candidates, payloads, min_score)[:top_k],
                timings={"embed_ms": embed_ms, "search_ms": search_ms, "rerank_ms": rerank_ms},
                warning="reranker_unavailable",
            )
        rerank_ms = self._millis(rerank_start)
        return SearchOutcome(
            chunks=self._build_reranked(candidates, payloads, scores, min_score)[:top_k],
            timings={"embed_ms": embed_ms, "search_ms": search_ms, "rerank_ms": rerank_ms},
            warning=None,
        )

    @staticmethod
    def _millis(start: float) -> float:
        return round((time.perf_counter() - start) * 1000.0, 3)

    def _build_reranked(
        self,
        candidates: list[tuple[str, float]],
        payloads: dict[str, dict],  # type: ignore[type-arg]
        scores: list[float],
        min_score: float | None,
    ) -> list[RetrievedChunk]:
        entries = [
            (embed_score, score, payloads.get(point_id, {}))
            for (point_id, embed_score), score in zip(candidates, scores, strict=True)
        ]
        entries.sort(key=lambda entry: entry[1], reverse=True)
        chunks: list[RetrievedChunk] = []
        for embed_score, rerank_score, payload in entries:
            if min_score is not None and rerank_score < min_score:
                continue
            chunks.append(
                RetrievedChunk(
                    content=payload.get("content", ""),
                    metadata=payload,
                    embed_score=embed_score,
                    rerank_score=rerank_score,
                )
            )
        return chunks

    def _build_degraded(
        self,
        candidates: list[tuple[str, float]],
        payloads: dict[str, dict],  # type: ignore[type-arg]
        min_score: float | None,
    ) -> list[RetrievedChunk]:
        entries = sorted(
            [(embed_score, payloads.get(point_id, {})) for point_id, embed_score in candidates],
            key=lambda entry: entry[0],
            reverse=True,
        )
        chunks: list[RetrievedChunk] = []
        for embed_score, payload in entries:
            if min_score is not None and embed_score < min_score:
                continue
            chunks.append(
                RetrievedChunk(
                    content=payload.get("content", ""),
                    metadata=payload,
                    embed_score=embed_score,
                    rerank_score=None,
                )
            )
        return chunks
