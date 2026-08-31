from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import uuid4

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import JobNotFoundError
from scholar_rag.core.types import JobRecord
from scholar_rag.store.layout import jobs_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, type TEXT, kb TEXT, params_json TEXT,
  status TEXT CHECK(status IN ('pending','running','succeeded','failed','cancelled','interrupted')),
  done INTEGER DEFAULT 0, total INTEGER DEFAULT 0, current_doc TEXT,
  error TEXT, result_json TEXT, created_at TEXT, updated_at TEXT);
"""

_HANDLER_TYPE = Callable[[str, dict], dict]  # type: ignore[type-arg]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class JobManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.load()
        db_path = jobs_db_path(self._settings)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        self._write_lock = threading.Lock()
        self._handlers: dict[str, _HANDLER_TYPE] = {}
        self._handlers_guard = threading.Lock()
        workers = max(1, self._settings.job_workers)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _get_row(self, job_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            row: sqlite3.Row | None = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return row

    def _execute(self, sql: str, params: tuple[object, ...]) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(sql, params)

    def submit(self, type: str, kb: str, params: dict) -> str:  # type: ignore[type-arg]
        job_id = uuid4().hex
        now = _now_iso()
        self._execute(
            "INSERT INTO jobs(job_id, type, kb, params_json, status, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (job_id, type, kb, json.dumps(params), now, now),
        )
        self._executor.submit(self._run, job_id)
        return job_id

    def _run(self, job_id: str) -> None:
        row = self._get_row(job_id)
        if row is None or row["status"] != "pending":
            return
        self._mark_running(job_id)
        handler = self._handler(row["type"])
        if handler is None:
            self._finish_failed(job_id, f"no handler registered for job type: {row['type']}")
            return
        try:
            params = json.loads(row["params_json"])
            result = handler(job_id, params)
        except Exception as exc:
            self._finish_failed(job_id, str(exc))
            return
        self._finish_succeeded(job_id, result)

    def _mark_running(self, job_id: str) -> None:
        self._execute(
            "UPDATE jobs SET status = 'running', updated_at = ? WHERE job_id = ?",
            (_now_iso(), job_id),
        )

    def _finish_succeeded(self, job_id: str, result: dict) -> None:  # type: ignore[type-arg]
        self._execute(
            "UPDATE jobs SET status = 'succeeded', result_json = ?, updated_at = ?"
            " WHERE job_id = ? AND status = 'running'",
            (json.dumps(result), _now_iso(), job_id),
        )

    def _finish_failed(self, job_id: str, error: str) -> None:
        self._execute(
            "UPDATE jobs SET status = 'failed', error = ?, updated_at = ?"
            " WHERE job_id = ? AND status = 'running'",
            (error, _now_iso(), job_id),
        )

    def _handler(self, type: str) -> _HANDLER_TYPE | None:
        with self._handlers_guard:
            return self._handlers.get(type)

    def register_handler(self, type: str, handler: Callable[[str, dict], dict]) -> None:  # type: ignore[type-arg]
        with self._handlers_guard:
            self._handlers[type] = handler

    def update_progress(self, job_id: str, done: int, current_doc: str) -> None:
        self._execute(
            "UPDATE jobs SET done = ?, current_doc = ?, updated_at = ? WHERE job_id = ?",
            (done, current_doc, _now_iso(), job_id),
        )

    def set_total(self, job_id: str, total: int) -> None:
        self._execute(
            "UPDATE jobs SET total = ?, updated_at = ? WHERE job_id = ?",
            (total, _now_iso(), job_id),
        )

    def delete_by_kb(self, kb: str) -> int:
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE kb = ?", (kb,))
            return cursor.rowcount

    def get(self, job_id) -> JobRecord:  # type: ignore[no-untyped-def]
        row = self._get_row(job_id)
        if row is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        return JobRecord(
            job_id=row["job_id"],
            type=row["type"],
            kb=row["kb"],
            status=row["status"],
            done=row["done"],
            total=row["total"],
            current_doc=row["current_doc"],
            error=row["error"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def recover_interrupted(self) -> int:
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
                (_now_iso(),),
            )
            return cursor.rowcount

    def close(self) -> None:
        self._executor.shutdown(wait=False)
