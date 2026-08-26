from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchChunksParams(BaseModel):
    kb: str
    query: str
    top_k: int = Field(10, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=0, le=1)
    metadata_filter: dict[str, Any] | None = None


class SearchDocumentsParams(BaseModel):
    kb: str
    query: str | None = None
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(10, ge=1, le=20)


class ListDocumentsParams(BaseModel):
    kb: str
    sort: str = "added"
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=20)


class GetDocumentParams(BaseModel):
    kb: str
    doc_id: str


class GetDocumentTextParams(BaseModel):
    kb: str
    doc_id: str
    section: str | None = None
    page: int = Field(1, ge=1)
    page_size: int = Field(6000, ge=1, le=8000)


class AddDocumentParams(BaseModel):
    kb: str
    pdf_path: str


class RemoveDocumentParams(BaseModel):
    kb: str
    doc_id: str


class CreateKbParams(BaseModel):
    kb_name: str
    folder_path: str
    chunk_min: int = 300
    chunk_max: int = 1500
    chunk_overlap: int = 100
    skip_existing: bool = True


class DeleteKbParams(BaseModel):
    kb: str
    confirm_token: str | None = None


class ListKbsParams(BaseModel):
    pass


class GetJobParams(BaseModel):
    job_id: str


class ChunkResult(BaseModel):
    content: str
    metadata: dict[str, Any]
    embed_score: float
    rerank_score: float | None


class SearchChunksResult(BaseModel):
    chunks: list[ChunkResult]
    timings: dict[str, float]
    warning: str | None


class DocumentRow(BaseModel):
    doc_id: str
    title: str
    year: int | None = None
    journal: str | None = None
    first_author: str = ""
    doi: str | None = None


class SearchDocumentsResult(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[DocumentRow]


class ListDocumentsResult(BaseModel):
    total: int
    page: int
    page_size: int
    results: list[DocumentRow]


class OutlineEntry(BaseModel):
    section: str
    char_count: int


class GetDocumentResult(BaseModel):
    doc_id: str
    metadata: dict[str, Any]
    abstract: str | None
    outline: list[OutlineEntry]
    total_chars: int
    usage_hint: str


class GetDocumentTextResult(BaseModel):
    content: str
    page: int
    total_pages: int
    total_chars: int
    next_hint: str | None


class AddDocumentResult(BaseModel):
    job_id: str


class RemoveDocumentResult(BaseModel):
    doc_id: str
    chunks_deleted: int
    catalog_deleted: bool
    files_deleted: bool


class CreateKbResult(BaseModel):
    job_id: str


class DeleteKbResult(BaseModel):
    kb: str | None = None
    doc_count: int | None = None
    chunk_count: int | None = None
    created_at: str | None = None
    confirm_token: str | None = None
    expires_in: int | None = None
    deleted: bool | None = None


class ListKbsResult(BaseModel):
    kbs: list[dict[str, Any]]


class GetJobResult(BaseModel):
    job_id: str
    type: str
    kb: str
    status: str
    progress: dict[str, int]
    current_doc: str | None
    error: str | None
    result_summary: dict[str, Any] | None
    timings: dict[str, float]
    created_at: str
    updated_at: str
