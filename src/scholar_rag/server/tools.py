from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from scholar_rag.core.errors import DocNotFoundError, KbNotFoundError, ScholarRagError
from scholar_rag.core.jobs import JobManager
from scholar_rag.core.operations import Operations
from scholar_rag.core.registry import Registry
from scholar_rag.retrieve import RetrievalEngine
from scholar_rag.server.schemas import (
    AddDocumentParams,
    AddDocumentResult,
    ChunkResult,
    CreateKbParams,
    CreateKbResult,
    DeleteKbParams,
    DeleteKbResult,
    DocumentRow,
    GetDocumentParams,
    GetDocumentResult,
    GetDocumentTextParams,
    GetDocumentTextResult,
    GetJobParams,
    GetJobResult,
    ListDocumentsParams,
    ListDocumentsResult,
    ListKbsParams,
    ListKbsResult,
    OutlineEntry,
    RemoveDocumentParams,
    RemoveDocumentResult,
    SearchChunksParams,
    SearchChunksResult,
    SearchDocumentsParams,
    SearchDocumentsResult,
)
from scholar_rag.store.catalog import Catalog
from scholar_rag.store.layout import catalog_path, documents_dir

TOOL_NAMES: tuple[str, ...] = (
    "search_chunks",
    "search_documents",
    "list_documents",
    "get_document",
    "get_document_text",
    "add_document",
    "remove_document",
    "create_kb",
    "delete_kb",
    "list_kbs",
    "get_job",
)

_USAGE_HINT = (
    "Use get_document_text with kb, doc_id, and optionally a section from the outline, "
    "plus page/page_size, to read the full text."
)

_HINTS: dict[str, str] = {
    "unknown_tool": "Available tools: " + ", ".join(TOOL_NAMES),
    "invalid_request": "Check the tool arguments (names, types and bounds) and retry.",
    "invalid_filter": "Allowed filter fields: doc_id, section, doi, journal, title, "
    "first_author, year, added_after.",
    "kb_not_found": "Create the kb first with create_kb.",
    "kb_exists": "Choose a different kb_name.",
    "doc_not_found": "Use list_documents to see valid doc_ids.",
    "doc_exists": "Remove the existing document first or use skip_existing.",
    "confirm_token_invalid": "Call delete_kb without confirm_token to obtain a fresh one.",
    "job_not_found": "Check job_id returned by create_kb or add_document.",
    "pipeline_stage_failed": "Check the uploaded file and the parse/embedding services.",
    "service_unavailable": "Check that qdrant and the model services are reachable.",
    "internal_error": "Retry; consult the server logs if it persists.",
}

_ops_instance: Operations | None = None
_ops_lock = threading.Lock()


def _operations() -> Operations:
    global _ops_instance
    with _ops_lock:
        if _ops_instance is None:
            manager = JobManager()
            instance = Operations(manager)
            instance.startup()
            _ops_instance = instance
    return _ops_instance


def _require_kb(kb: str) -> None:
    if not Registry().kb_path(kb).is_dir():
        raise KbNotFoundError(f"kb not found: {kb}")


def _validation_message(exc: ValidationError) -> str:
    locs: list[str] = []
    for error in exc.errors():
        loc = error.get("loc")
        if isinstance(loc, tuple) and loc:
            locs.append(".".join(str(part) for part in loc))
        if len(locs) >= 3:
            break
    if locs:
        return f"invalid arguments: {', '.join(locs)}"
    return "invalid arguments"


def _run_search_chunks(params: SearchChunksParams) -> dict[str, Any]:
    _require_kb(params.kb)
    outcome = RetrievalEngine(params.kb).search(
        params.query, params.top_k, params.min_score, params.metadata_filter
    )
    result = SearchChunksResult(
        chunks=[
            ChunkResult(
                content=chunk.content,
                metadata=chunk.metadata,
                embed_score=chunk.embed_score,
                rerank_score=chunk.rerank_score,
            )
            for chunk in outcome.chunks
        ],
        timings=outcome.timings,
        warning=outcome.warning,
    )
    return result.model_dump()


def _project_rows(rows: list[dict[str, Any]]) -> list[DocumentRow]:
    projected: list[DocumentRow] = []
    for row in rows:
        projected.append(
            DocumentRow(
                doc_id=row["doc_id"],
                title=row["title"],
                year=row.get("year"),
                journal=row.get("journal"),
                first_author=row.get("first_author") or "",
                doi=row.get("doi"),
            )
        )
    return projected


def _run_search_documents(params: SearchDocumentsParams) -> dict[str, Any]:
    _require_kb(params.kb)
    catalog = Catalog(catalog_path(params.kb))
    total, rows = catalog.search_documents(
        params.query,
        params.title,
        params.authors,
        params.journal,
        params.year_from,
        params.year_to,
        params.page,
        params.page_size,
    )
    result = SearchDocumentsResult(
        total=total,
        page=params.page,
        page_size=params.page_size,
        results=_project_rows(rows),
    )
    return result.model_dump()


def _run_list_documents(params: ListDocumentsParams) -> dict[str, Any]:
    _require_kb(params.kb)
    catalog = Catalog(catalog_path(params.kb))
    total, rows = catalog.list_documents(params.sort, params.page, params.page_size)
    result = ListDocumentsResult(
        total=total,
        page=params.page,
        page_size=params.page_size,
        results=_project_rows(rows),
    )
    return result.model_dump()


def _read_sections_payload(kb: str, doc_id: str) -> dict[str, Any]:
    path = documents_dir(kb) / doc_id / "sections.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid sections.json payload")
    return payload


def _run_get_document(params: GetDocumentParams) -> dict[str, Any]:
    _require_kb(params.kb)
    catalog = Catalog(catalog_path(params.kb))
    record = catalog.get_document(params.doc_id)
    if record is None:
        raise DocNotFoundError(f"document not found: {params.doc_id}")
    payload = _read_sections_payload(params.kb, params.doc_id)
    outline = [
        OutlineEntry(section=name, char_count=int(count)) for name, count in payload["sections"]
    ]
    result = GetDocumentResult(
        doc_id=params.doc_id,
        metadata=record,
        abstract=record.get("abstract"),
        outline=outline,
        total_chars=int(payload["total_chars"]),
        usage_hint=_USAGE_HINT,
    )
    return result.model_dump()


def _section_text(payload: dict[str, Any], section: str, available: list[str]) -> str:
    texts = payload.get("section_texts")
    if isinstance(texts, dict) and section in texts:
        return str(texts[section])
    listing = ", ".join(available) if available else "(none)"
    raise ValueError(f"section not found: {section}; available sections: {listing}")


def _run_get_document_text(params: GetDocumentTextParams) -> dict[str, Any]:
    _require_kb(params.kb)
    catalog = Catalog(catalog_path(params.kb))
    if catalog.get_document(params.doc_id) is None:
        raise DocNotFoundError(f"document not found: {params.doc_id}")
    doc_dir = documents_dir(params.kb) / params.doc_id
    if params.section is None:
        text = (doc_dir / "full_text.md").read_text(encoding="utf-8")
    else:
        payload = _read_sections_payload(params.kb, params.doc_id)
        available = [str(entry[0]) for entry in payload["sections"]]
        text = _section_text(payload, params.section, available)
    total_pages = max(1, math.ceil(len(text) / params.page_size))
    if params.page > total_pages:
        raise ValueError(f"page {params.page} out of range, pages 1..{total_pages}")
    start = (params.page - 1) * params.page_size
    content = text[start : start + params.page_size]
    next_hint = f"next page: {params.page + 1}" if params.page < total_pages else None
    result = GetDocumentTextResult(
        content=content,
        page=params.page,
        total_pages=total_pages,
        total_chars=len(text),
        next_hint=next_hint,
    )
    return result.model_dump()


def _run_add_document(params: AddDocumentParams) -> dict[str, Any]:
    job_id = _operations().add_document(params.kb, params.pdf_path)
    return AddDocumentResult(job_id=job_id).model_dump()


def _run_remove_document(params: RemoveDocumentParams) -> dict[str, Any]:
    outcome = _operations().remove_document(params.kb, params.doc_id)
    return RemoveDocumentResult(**outcome).model_dump()


def _run_create_kb(params: CreateKbParams) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "chunk_min": params.chunk_min,
        "chunk_max": params.chunk_max,
        "chunk_overlap": params.chunk_overlap,
        "skip_existing": params.skip_existing,
    }
    job_id = _operations().create_kb(params.kb_name, params.folder_path, opts)
    return CreateKbResult(job_id=job_id).model_dump()


def _run_delete_kb(params: DeleteKbParams) -> dict[str, Any]:
    outcome = _operations().delete_kb(params.kb, params.confirm_token)
    return DeleteKbResult(**outcome).model_dump(exclude_none=True)


def _run_list_kbs(params: ListKbsParams) -> dict[str, Any]:
    return ListKbsResult(kbs=_operations().list_kbs()).model_dump()


def _run_get_job(params: GetJobParams) -> dict[str, Any]:
    record = _operations().get_job(params.job_id)
    result = GetJobResult(
        job_id=record.job_id,
        type=record.type,
        kb=record.kb,
        status=record.status,
        progress={"done": record.done, "total": record.total},
        current_doc=record.current_doc,
        error=record.error,
        result_summary=record.result,
        timings={},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
    return result.model_dump()


_IMPLS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "search_chunks": _run_search_chunks,
    "search_documents": _run_search_documents,
    "list_documents": _run_list_documents,
    "get_document": _run_get_document,
    "get_document_text": _run_get_document_text,
    "add_document": _run_add_document,
    "remove_document": _run_remove_document,
    "create_kb": _run_create_kb,
    "delete_kb": _run_delete_kb,
    "list_kbs": _run_list_kbs,
    "get_job": _run_get_job,
}

_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "search_chunks": SearchChunksParams,
    "search_documents": SearchDocumentsParams,
    "list_documents": ListDocumentsParams,
    "get_document": GetDocumentParams,
    "get_document_text": GetDocumentTextParams,
    "add_document": AddDocumentParams,
    "remove_document": RemoveDocumentParams,
    "create_kb": CreateKbParams,
    "delete_kb": DeleteKbParams,
    "list_kbs": ListKbsParams,
    "get_job": GetJobParams,
}


def _error(code: str, message: str, hint: str) -> dict[str, str]:
    return {"error_code": code, "message": message, "hint": hint}


def dispatch_tool(name: str, arguments: Any) -> dict[str, Any]:
    impl = _IMPLS.get(name)
    model = _PARAM_MODELS.get(name)
    if impl is None or model is None:
        return _error("unknown_tool", f"unknown tool: {name}", _HINTS["unknown_tool"])
    if not isinstance(arguments, dict):
        return _error(
            "invalid_request",
            "arguments must be an object of named parameters",
            _HINTS["invalid_request"],
        )
    try:
        params = model(**arguments)
    except ValidationError as exc:
        return _error("invalid_request", _validation_message(exc), _HINTS["invalid_request"])
    try:
        return impl(params)
    except ScholarRagError as exc:
        return _error(exc.code, str(exc), _HINTS.get(exc.code, _HINTS["internal_error"]))
    except ValueError as exc:
        return _error("invalid_request", str(exc), _HINTS["invalid_request"])
    except Exception:
        return _error(
            "internal_error",
            "internal error",
            _HINTS["internal_error"],
        )


def tool_search_chunks(
    kb: str,
    query: str,
    top_k: int = 10,
    min_score: float | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return dispatch_tool(
        "search_chunks",
        {
            "kb": kb,
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "metadata_filter": metadata_filter,
        },
    )


def tool_search_documents(
    kb: str,
    query: str | None = None,
    title: str | None = None,
    authors: str | None = None,
    journal: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    page: int = 1,
    page_size: int = 10,
) -> dict[str, Any]:
    return dispatch_tool(
        "search_documents",
        {
            "kb": kb,
            "query": query,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year_from": year_from,
            "year_to": year_to,
            "page": page,
            "page_size": page_size,
        },
    )


def tool_list_documents(
    kb: str,
    sort: str = "added",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    return dispatch_tool(
        "list_documents",
        {"kb": kb, "sort": sort, "page": page, "page_size": page_size},
    )


def tool_get_document(kb: str, doc_id: str) -> dict[str, Any]:
    return dispatch_tool("get_document", {"kb": kb, "doc_id": doc_id})


def tool_get_document_text(
    kb: str,
    doc_id: str,
    section: str | None = None,
    page: int = 1,
    page_size: int = 6000,
) -> dict[str, Any]:
    return dispatch_tool(
        "get_document_text",
        {"kb": kb, "doc_id": doc_id, "section": section, "page": page, "page_size": page_size},
    )


def tool_add_document(kb: str, pdf_path: str) -> dict[str, Any]:
    return dispatch_tool("add_document", {"kb": kb, "pdf_path": pdf_path})


def tool_remove_document(kb: str, doc_id: str) -> dict[str, Any]:
    return dispatch_tool("remove_document", {"kb": kb, "doc_id": doc_id})


def tool_create_kb(
    kb_name: str,
    folder_path: str,
    chunk_min: int = 300,
    chunk_max: int = 1500,
    chunk_overlap: int = 100,
    skip_existing: bool = True,
) -> dict[str, Any]:
    return dispatch_tool(
        "create_kb",
        {
            "kb_name": kb_name,
            "folder_path": folder_path,
            "chunk_min": chunk_min,
            "chunk_max": chunk_max,
            "chunk_overlap": chunk_overlap,
            "skip_existing": skip_existing,
        },
    )


def tool_delete_kb(kb: str, confirm_token: str | None = None) -> dict[str, Any]:
    return dispatch_tool("delete_kb", {"kb": kb, "confirm_token": confirm_token})


def tool_list_kbs() -> dict[str, Any]:
    return dispatch_tool("list_kbs", {})


def tool_get_job(job_id: str) -> dict[str, Any]:
    return dispatch_tool("get_job", {"job_id": job_id})
