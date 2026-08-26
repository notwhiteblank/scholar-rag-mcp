from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetadataResult:
    title: str
    year: int | None
    journal: str | None
    doi: str | None
    authors: list[str]
    abstract: str | None
    source: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnotatedBlock:
    text: str
    is_heading: bool
    heading_level: int
    section: str
    confidence: float


@dataclass(frozen=True)
class AnnotatedDocument:
    blocks: tuple[AnnotatedBlock, ...]
    sections: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    content: str
    section: str
    heading_path: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    embed_score: float
    rerank_score: float | None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    type: str
    kb: str
    status: str
    done: int
    total: int
    current_doc: str | None
    error: str | None
    result: dict[str, Any] | None
    created_at: str
    updated_at: str
