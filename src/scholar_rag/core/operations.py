from __future__ import annotations

import contextlib
import json
import secrets
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import ConfirmTokenError, DocNotFoundError, KbNotFoundError
from scholar_rag.core.jobs import JobManager
from scholar_rag.core.registry import Registry
from scholar_rag.core.types import JobRecord
from scholar_rag.ingest.pipeline import ingest_document
from scholar_rag.ingest.staging import ChunkConfig
from scholar_rag.models import get_embedding_client
from scholar_rag.store.catalog import Catalog
from scholar_rag.store.layout import catalog_path, documents_dir
from scholar_rag.store.qdrant_manager import get_qdrant_manager
from scholar_rag.store.vector_store import VectorStore

_TOKEN_TTL_SECONDS = 600


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scan_pdf_files(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"folder not found: {folder}")
    pdfs = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    if not pdfs:
        raise ValueError("no pdf files found")
    return pdfs


class Operations:
    def __init__(self, job_manager: JobManager) -> None:
        self._jobs = job_manager
        self._registry = Registry()
        self._tokens: dict[str, tuple[str, float]] = {}
        self._tokens_guard = threading.Lock()
        self._jobs.register_handler("create_kb", self._run_create_kb)
        self._jobs.register_handler("add_document", self._run_add_document)

    def _ensure_kb(self, kb: str) -> None:
        if not self._registry.kb_path(kb).is_dir():
            raise KbNotFoundError(f"kb not found: {kb}")

    def _read_chunk_opts(self, opts: dict) -> tuple[int, int, int, bool]:  # type: ignore[type-arg]
        return (
            int(opts.get("chunk_min", 300)),
            int(opts.get("chunk_max", 1500)),
            int(opts.get("chunk_overlap", 100)),
            bool(opts.get("skip_existing", True)),
        )

    def _kb_chunk_config(self, kb: str) -> ChunkConfig:
        meta_path = self._registry.kb_path(kb) / "kb_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk = meta.get("chunk") or {}
        return ChunkConfig(
            min_chunk=int(chunk.get("min", 300)),
            max_chunk=int(chunk.get("max", 1500)),
            overlap=int(chunk.get("overlap", 100)),
        )

    def create_kb(self, name: str, folder: str, opts: dict | None = None) -> str:  # type: ignore[type-arg]
        self._registry.validate_name(name)
        _scan_pdf_files(folder)
        opts_dict = opts or {}
        min_chunk, max_chunk, overlap, _skip = self._read_chunk_opts(opts_dict)
        settings = Settings.load()
        dim = len(get_embedding_client().embed_query("dimension probe"))
        meta = {
            "name": name,
            "created_at": _now_iso(),
            "dim": dim,
            "embedding_model": settings.embed_model,
            "chunk": {"min": min_chunk, "max": max_chunk, "overlap": overlap},
            "schema_version": 1,
        }
        self._registry.create_kb(name, meta)
        try:
            VectorStore.create(name, dim)
        except Exception:
            with contextlib.suppress(Exception):
                get_qdrant_manager().client().delete_collection(f"kb_{name}")
            with contextlib.suppress(Exception):
                self._registry.delete_kb(name)
            raise
        return self._jobs.submit("create_kb", name, {"folder": folder, "opts": opts_dict})

    def _run_create_kb(self, job_id: str, params: dict) -> dict:  # type: ignore[type-arg]
        kb = self._jobs.get(job_id).kb
        folder = params["folder"]
        opts = params.get("opts") or {}
        min_chunk, max_chunk, overlap, skip_existing = self._read_chunk_opts(opts)
        config = ChunkConfig(min_chunk=min_chunk, max_chunk=max_chunk, overlap=overlap)
        pdfs = _scan_pdf_files(folder)
        total = len(pdfs)
        self._jobs.set_total(job_id, total)
        lock = self._registry.get_kb_lock(kb)
        succeeded: list[str] = []
        skipped = 0
        failures: list[dict[str, str]] = []
        with lock:
            for index, pdf in enumerate(pdfs, start=1):
                try:
                    outcome = ingest_document(pdf, kb, config, skip_existing=skip_existing)
                except Exception as exc:
                    failures.append({"doc": str(pdf), "error": str(exc)})
                else:
                    if outcome.get("skipped"):
                        skipped += 1
                    else:
                        succeeded.append(str(outcome["doc_id"]))
                self._jobs.update_progress(job_id, index, str(pdf))
        return {
            "total": total,
            "succeeded": len(succeeded),
            "failed": len(failures),
            "skipped": skipped,
            "failures": failures,
        }

    def add_document(self, kb: str, pdf_path: str) -> str:
        self._ensure_kb(kb)
        if not Path(pdf_path).is_file():
            raise ValueError(f"pdf not found: {pdf_path}")
        return self._jobs.submit("add_document", kb, {"pdf_path": pdf_path})

    def _run_add_document(self, job_id: str, params: dict) -> dict:  # type: ignore[type-arg]
        kb = self._jobs.get(job_id).kb
        config = self._kb_chunk_config(kb)
        pdf = Path(params["pdf_path"])
        self._jobs.set_total(job_id, 1)
        lock = self._registry.get_kb_lock(kb)
        with lock:
            outcome = ingest_document(pdf, kb, config, skip_existing=False)
        return {
            "doc_id": outcome["doc_id"],
            "title": outcome["title"],
            "chunks": outcome["chunks"],
            "skipped": outcome["skipped"],
        }

    def remove_document(self, kb: str, doc_id: str) -> dict:  # type: ignore[type-arg]
        self._ensure_kb(kb)
        catalog = Catalog(catalog_path(kb))
        if catalog.get_document(doc_id) is None:
            raise DocNotFoundError(f"document not found: {doc_id}")
        lock = self._registry.get_kb_lock(kb)
        with lock:
            deleted_points = VectorStore.open(kb).delete_doc(doc_id)
            catalog.delete_document(doc_id)
            doc_dir = documents_dir(kb) / doc_id
            files_deleted = False
            if doc_dir.is_dir():
                shutil.rmtree(doc_dir)
                files_deleted = True
        return {
            "doc_id": doc_id,
            "chunks_deleted": deleted_points,
            "catalog_deleted": True,
            "files_deleted": files_deleted,
        }

    def delete_kb(self, kb: str, confirm_token: str | None = None) -> dict:  # type: ignore[type-arg]
        self._ensure_kb(kb)
        if confirm_token is None:
            return self._delete_kb_preview(kb)
        self._validate_token(kb, confirm_token)
        lock = self._registry.get_kb_lock(kb)
        with lock:
            VectorStore.open(kb).drop()
            self._registry.delete_kb(kb)
            self._jobs.delete_by_kb(kb)
            self._clear_token(kb)
        return {"kb": kb, "deleted": True}

    def _delete_kb_preview(self, kb: str) -> dict[str, object]:
        meta_path = self._registry.kb_path(kb) / "kb_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_count = Catalog(catalog_path(kb)).count_documents()
        chunk_count = VectorStore.open(kb).count()
        token = secrets.token_hex(16)
        expires_at = time.time() + _TOKEN_TTL_SECONDS
        with self._tokens_guard:
            self._tokens[kb] = (token, expires_at)
        return {
            "kb": kb,
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "created_at": meta.get("created_at"),
            "confirm_token": token,
            "expires_in": _TOKEN_TTL_SECONDS,
        }

    def _validate_token(self, kb: str, token: str) -> None:
        with self._tokens_guard:
            entry = self._tokens.get(kb)
        if entry is None:
            raise ConfirmTokenError(f"no confirm token issued for kb: {kb}")
        expected, expires_at = entry
        if time.time() > expires_at:
            raise ConfirmTokenError(f"confirm token expired for kb: {kb}")
        if token != expected:
            raise ConfirmTokenError(f"confirm token mismatch for kb: {kb}")

    def _clear_token(self, kb: str) -> None:
        with self._tokens_guard:
            self._tokens.pop(kb, None)

    def list_kbs(self) -> list[dict]:  # type: ignore[type-arg]
        return self._registry.list_kbs()

    def get_job(self, job_id) -> JobRecord:  # type: ignore[no-untyped-def]
        return self._jobs.get(job_id)

    def startup(self) -> int:
        return self._jobs.recover_interrupted()
