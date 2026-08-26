from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.core.types import Chunk
from scholar_rag.store.qdrant_manager import get_qdrant_manager

_BATCH_SIZE = 256
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "journal_norm": models.PayloadSchemaType.KEYWORD,
    "title_norm": models.PayloadSchemaType.KEYWORD,
    "first_author_norm": models.PayloadSchemaType.KEYWORD,
    "doi": models.PayloadSchemaType.KEYWORD,
    "added_ts": models.PayloadSchemaType.FLOAT,
}


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}#{chunk_index}"))


class VectorStore:
    def __init__(self, client: QdrantClient, kb: str) -> None:
        self._client = client
        self._kb = kb
        self._collection = f"kb_{kb}"

    @classmethod
    def create(cls, kb: str, dim: int) -> VectorStore:
        client = get_qdrant_manager().client()
        store = cls(client, kb)
        client.create_collection(
            collection_name=store._collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        for field, schema in _PAYLOAD_INDEXES.items():
            client.create_payload_index(
                collection_name=store._collection, field_name=field, field_schema=schema
            )
        return store

    @classmethod
    def open(cls, kb: str) -> VectorStore:
        client = get_qdrant_manager().client()
        store = cls(client, kb)
        if not client.collection_exists(store._collection):
            raise ServiceUnavailableError(f"vector collection {store._collection} not found")
        return store

    def upsert_chunks(
        self, doc_id: str, chunks: list[Chunk], embeds: list[list[float]], payload_base: dict  # type: ignore[type-arg]
    ) -> list[str]:
        if len(chunks) != len(embeds):
            raise ValueError("chunks and embeds length mismatch")
        points: list[models.PointStruct] = []
        for chunk, embed in zip(chunks, embeds, strict=True):
            payload = dict(payload_base)
            payload["chunk_index"] = chunk.chunk_index
            payload["section"] = chunk.section
            payload["heading_path"] = chunk.heading_path
            points.append(
                models.PointStruct(
                    id=_point_id(doc_id, chunk.chunk_index), vector=embed, payload=payload
                )
            )
        point_ids = [str(point.id) for point in points]
        for start in range(0, len(points), _BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection,
                points=points[start : start + _BATCH_SIZE],
                wait=False,
            )
        return point_ids

    def query(
        self, vector: list[float], qfilter: Any | None, limit: int
    ) -> list[tuple[str, float]]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=qfilter,
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )
        return [(str(point.id), point.score) for point in result.points]

    def payloads(self, point_ids: list[str]) -> dict[str, dict]:  # type: ignore[type-arg]
        records = self._client.retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return {str(record.id): record.payload or {} for record in records}

    def delete_doc(self, doc_id: str) -> int:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        total = self._client.count(
            collection_name=self._collection, count_filter=query_filter
        ).count
        self._client.delete(
            collection_name=self._collection, points_selector=query_filter, wait=True
        )
        return total

    def drop(self) -> None:
        self._client.delete_collection(collection_name=self._collection)

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
