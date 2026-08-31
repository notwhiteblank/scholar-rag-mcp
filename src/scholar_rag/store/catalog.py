from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from scholar_rag.core.types import MetadataResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER,
  journal TEXT, doi TEXT, first_author TEXT, author_count INTEGER,
  source_path TEXT, content_hash TEXT UNIQUE, added_at TEXT, parse_status TEXT);
CREATE TABLE IF NOT EXISTS document_authors(doc_id TEXT, position INTEGER, name TEXT, PRIMARY KEY(doc_id, position));
CREATE TABLE IF NOT EXISTS document_keywords(doc_id TEXT, keyword TEXT, PRIMARY KEY(doc_id, keyword));
CREATE TABLE IF NOT EXISTS chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, section TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(doc_id UNINDEXED, title, abstract, keywords, tokenize='trigram');
"""

_ROW_FIELDS = ("doc_id", "title", "year", "journal", "first_author", "doi", "added_at")
_SORTS = {"added": "added_at DESC", "title": "title ASC", "year": "year ASC"}
_PARSE_STATUS_DONE = "done"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts_query(query: str) -> str:
    words = [token.replace('"', "") for token in query.split()]
    words = [word for word in words if word]
    return " ".join('"' + word + '"' for word in words)


class Catalog:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _delete_rows(self, conn: sqlite3.Connection, doc_id: str) -> None:
        for table in ("document_authors", "document_keywords", "chunks", "documents"):
            conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents_fts WHERE doc_id = ?", (doc_id,))

    def _upsert_rows(self, conn: sqlite3.Connection, meta: MetadataResult, doc_id: str, source_path: str, content_hash: str) -> None:
        self._delete_rows(conn, doc_id)
        conn.execute(
            "INSERT INTO documents(doc_id, title, abstract, year, journal, doi, first_author,"
            " author_count, source_path, content_hash, added_at, parse_status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                meta.title,
                meta.abstract,
                meta.year,
                meta.journal,
                meta.doi,
                meta.authors[0] if meta.authors else "",
                len(meta.authors),
                source_path,
                content_hash,
                _now_iso(),
                _PARSE_STATUS_DONE,
            ),
        )
        conn.executemany(
            "INSERT INTO document_authors(doc_id, position, name) VALUES (?, ?, ?)",
            [(doc_id, position, name) for position, name in enumerate(meta.authors)],
        )
        conn.executemany(
            "INSERT INTO document_keywords(doc_id, keyword) VALUES (?, ?)",
            [(doc_id, keyword) for keyword in meta.keywords],
        )
        conn.execute(
            "INSERT INTO documents_fts(doc_id, title, abstract, keywords) VALUES (?, ?, ?, ?)",
            (doc_id, meta.title, meta.abstract or "", " ".join(meta.keywords)),
        )

    def upsert_document(
        self, meta: MetadataResult, doc_id: str, source_path: str, content_hash: str
    ) -> None:
        with self._connect() as conn:
            self._upsert_rows(conn, meta, doc_id, source_path, content_hash)

    def add_document(
        self, meta: MetadataResult, doc_id: str, source_path: str, content_hash: str, chunk_ids: list[str]
    ) -> None:
        with self._connect() as conn:
            self._upsert_rows(conn, meta, doc_id, source_path, content_hash)
            self._insert_chunk_rows(conn, doc_id, chunk_ids)

    def find_by_hash(self, content_hash) -> dict | None:  # type: ignore[type-arg, no-untyped-def]
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_document(self, doc_id) -> dict | None:  # type: ignore[type-arg, no-untyped-def]
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["authors"] = [
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM document_authors WHERE doc_id = ? ORDER BY position",
                    (doc_id,),
                )
            ]
            result["keywords"] = [
                r["keyword"]
                for r in conn.execute(
                    "SELECT keyword FROM document_keywords WHERE doc_id = ? ORDER BY keyword",
                    (doc_id,),
                )
            ]
            return result

    def list_documents(self, sort: str, page: int, page_size: int) -> tuple[int, list[dict]]:  # type: ignore[type-arg]
        if sort not in _SORTS:
            raise ValueError(f"invalid sort: {sort!r} (allowed: {', '.join(sorted(_SORTS))})")
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            rows = conn.execute(
                f"""SELECT {', '.join(_ROW_FIELDS)} FROM documents
                    ORDER BY {_SORTS[sort]} LIMIT ? OFFSET ?""",
                (page_size, offset),
            ).fetchall()
        return total, [dict(row) for row in rows]

    def search_documents(
        self,
        query: str | None,
        title: str | None,
        authors: str | None,
        journal: str | None,
        year_from: int | None,
        year_to: int | None,
        page: int,
        page_size: int,
    ) -> tuple[int, list[dict]]:  # type: ignore[type-arg]
        conditions: list[str] = []
        params: list[object] = []
        if query:
            fts_expr = _fts_query(query)
            if fts_expr:
                conditions.append("d.doc_id IN (SELECT doc_id FROM documents_fts WHERE documents_fts MATCH ?)")
                params.append(fts_expr)
        if title:
            conditions.append("d.title LIKE ? ESCAPE '\\'")
            params.append(f"%{_like_escape(title)}%")
        if authors:
            conditions.append(
                "EXISTS (SELECT 1 FROM document_authors a WHERE a.doc_id = d.doc_id"
                " AND a.name LIKE ? ESCAPE '\\')"
            )
            params.append(f"%{_like_escape(authors)}%")
        if journal:
            conditions.append("d.journal LIKE ? ESCAPE '\\'")
            params.append(f"%{_like_escape(journal)}%")
        if year_from is not None:
            conditions.append("d.year >= ?")
            params.append(year_from)
        if year_to is not None:
            conditions.append("d.year <= ?")
            params.append(year_to)
        where = " AND ".join(conditions) if conditions else "1=1"
        offset = (page - 1) * page_size
        with self._connect() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM documents d WHERE {where}", params).fetchone()["n"]
            rows = conn.execute(
                f"""SELECT d.doc_id, d.title, d.year, d.journal, d.first_author, d.doi, d.added_at
                    FROM documents d WHERE {where} ORDER BY d.added_at DESC LIMIT ? OFFSET ?""",
                (*params, page_size, offset),
            ).fetchall()
        return total, [dict(row) for row in rows]

    def insert_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        with self._connect() as conn:
            self._insert_chunk_rows(conn, doc_id, chunk_ids)

    def _insert_chunk_rows(self, conn: sqlite3.Connection, doc_id: str, chunk_ids: list[str]) -> None:
        rows = [(chunk_id, doc_id, None, "") for chunk_id in chunk_ids]
        conn.executemany(
            "INSERT INTO chunks(chunk_id, doc_id, chunk_index, section) VALUES (?, ?, ?, ?)",
            rows,
        )

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            self._delete_rows(conn, doc_id)

    def count_documents(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
