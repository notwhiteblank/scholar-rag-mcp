from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholar_rag.core.errors import DocExistsError, PipelineStageError
from scholar_rag.core.types import AnnotatedDocument, MetadataResult
from scholar_rag.ingest.annotate import stage_annotate
from scholar_rag.ingest.chunk import stage_chunks
from scholar_rag.ingest.clean import stage_clean
from scholar_rag.ingest.metadata import build_doc_id, stage_metadata
from scholar_rag.ingest.parse import stage_parse
from scholar_rag.ingest.staging import ChunkConfig, StagedPdf, stage_pdf
from scholar_rag.models import get_embedding_client
from scholar_rag.store.catalog import Catalog
from scholar_rag.store.layout import catalog_path, documents_dir, parse_cache_dir
from scholar_rag.store.vector_store import VectorStore


def _stage(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        raise PipelineStageError(f"{name} failed: {exc}", stage=name, cause=str(exc)) from exc


def _payload_base(meta: MetadataResult, doc_id: str, staged: StagedPdf) -> dict[str, Any]:
    now = datetime.now(UTC)
    first_author = meta.authors[0] if meta.authors else ""
    journal = meta.journal or ""
    return {
        "doc_id": doc_id,
        "title": meta.title,
        "title_norm": meta.title.lower(),
        "year": meta.year or 0,
        "journal": journal,
        "journal_norm": journal.lower(),
        "first_author": first_author,
        "first_author_norm": first_author.lower(),
        "author_count": len(meta.authors),
        "doi": meta.doi or "",
        "source_file": str(staged.source_path),
        "content_hash": staged.content_hash,
        "added_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "added_ts": now.timestamp(),
    }


def _persist_qdrant(kb: str, doc_id: str, chunks, embeds, payload_base) -> list[str]:  # type: ignore[no-untyped-def]
    return VectorStore.open(kb).upsert_chunks(doc_id, chunks, embeds, payload_base)


def _persist_catalog(
    catalog: Catalog, meta: MetadataResult, doc_id: str, staged: StagedPdf, point_ids: list[str]
) -> None:
    catalog.add_document(meta, doc_id, str(staged.source_path), staged.content_hash, point_ids)


def _persist_disk(
    persist_root: Path,
    work_dir: Path,
    doc_id: str,
    staged: StagedPdf,
    cleaned: str,
    annotated: AnnotatedDocument,
) -> None:
    source_target = work_dir / "source.pdf"
    source_target.write_bytes(staged.path.read_bytes())
    staged.path.unlink()
    (work_dir / "full_text.md").write_text(cleaned, encoding="utf-8")
    sections = [list(pair) for pair in annotated.sections]
    section_texts: dict[str, str] = {name: "" for name, _count in annotated.sections}
    for block in annotated.blocks:
        if not block.is_heading:
            section_texts[block.section] += block.text
    payload = {
        "sections": sections,
        "total_chars": sum(count for _name, count in annotated.sections),
        "section_texts": section_texts,
    }
    (work_dir / "sections.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    target = persist_root / doc_id
    if target.exists():
        raise OSError(f"document directory already exists: {target}")
    os.rename(work_dir, target)


def _cleanup_partial(kb: str, catalog: Catalog, doc_id: str) -> None:
    with contextlib.suppress(Exception):
        VectorStore.open(kb).delete_doc(doc_id)
    with contextlib.suppress(Exception):
        catalog.delete_document(doc_id)


def ingest_document(
    pdf_path: Path, kb: str, chunk_cfg: ChunkConfig, *, skip_existing: bool = False
) -> dict:  # type: ignore[type-arg]
    catalog = Catalog(catalog_path(kb))
    persist_root = documents_dir(kb)
    persist_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".ingest-", dir=str(persist_root)))
    doc_id: str | None = None
    try:
        staged = _stage("stage_pdf", lambda: stage_pdf(pdf_path, work_dir))
        existing = catalog.find_by_hash(staged.content_hash)
        if existing is not None:
            if skip_existing:
                return {
                    "doc_id": existing["doc_id"],
                    "title": existing["title"],
                    "chunks": 0,
                    "skipped": True,
                }
            raise DocExistsError(doc_id=existing["doc_id"])
        markdown = _stage("parse", lambda: stage_parse(staged, parse_cache_dir()))
        meta = _stage("metadata", lambda: stage_metadata(staged, markdown))
        doc_id = build_doc_id(meta, staged.source_path.name)
        cleaned = _stage("clean", lambda: stage_clean(markdown))
        annotated = _stage("annotate", lambda: stage_annotate(cleaned))
        chunks = _stage("chunks", lambda: stage_chunks(annotated, meta, chunk_cfg))
        embeds = _stage(
            "embed", lambda: get_embedding_client().embed([chunk.content for chunk in chunks])
        )
        payload_base = _payload_base(meta, doc_id, staged)
        point_ids = _stage(
            "persist_qdrant",
            lambda: _persist_qdrant(kb, doc_id, chunks, embeds, payload_base),
        )
        _stage(
            "persist_catalog",
            lambda: _persist_catalog(catalog, meta, doc_id, staged, point_ids),
        )
        _stage(
            "persist_disk",
            lambda: _persist_disk(persist_root, work_dir, doc_id, staged, cleaned, annotated),
        )
        return {"doc_id": doc_id, "title": meta.title, "chunks": len(chunks), "skipped": False}
    except Exception:
        if doc_id is not None:
            _cleanup_partial(kb, catalog, doc_id)
        raise
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
