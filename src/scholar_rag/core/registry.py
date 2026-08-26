from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import KbExistsError, KbNotFoundError
from scholar_rag.store.catalog import Catalog
from scholar_rag.store.layout import kb_dir, kbs_root

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_META_FILE = "kb_meta.json"
_CATALOG_FILE = "catalog.sqlite3"
_DOCUMENTS_DIR = "documents"


class Registry:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.load()
        self._root = kbs_root(self._settings)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def list_kbs(self) -> list[dict]:  # type: ignore[type-arg]
        if not self._root.is_dir():
            return []
        result: list[dict[str, Any]] = []
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            result.append(self._inspect_kb(entry))
        return result

    def _inspect_kb(self, directory: Path) -> dict[str, Any]:
        base: dict[str, Any] = {"name": directory.name}
        try:
            meta = json.loads((directory / _META_FILE).read_text(encoding="utf-8"))
            catalog = Catalog(directory / _CATALOG_FILE)
            return {
                **base,
                "doc_count": catalog.count_documents(),
                "created_at": meta.get("created_at"),
                "status": "ready",
            }
        except (OSError, ValueError, json.JSONDecodeError):
            return {**base, "doc_count": 0, "created_at": None, "status": "broken"}

    def create_kb(self, name: str, meta: dict) -> None:  # type: ignore[type-arg]
        self.validate_name(name)
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._root / name
        if target.exists():
            raise KbExistsError(f"kb already exists: {name}")
        tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=str(self._root)))
        try:
            (tmp / _META_FILE).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            Catalog(tmp / _CATALOG_FILE)
            (tmp / _DOCUMENTS_DIR).mkdir()
            os.rename(tmp, target)
        except FileExistsError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise KbExistsError(f"kb already exists: {name}") from exc
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def delete_kb(self, name: str) -> None:
        self.validate_name(name)
        target = self._root / name
        if not target.is_dir():
            raise KbNotFoundError(f"kb not found: {name}")
        shutil.rmtree(target)
        with self._locks_guard:
            self._locks.pop(name, None)

    def kb_path(self, name: str) -> Path:
        self.validate_name(name)
        return kb_dir(name, self._settings)

    def validate_name(self, name: str) -> None:
        if _NAME_RE.fullmatch(name) is None:
            raise ValueError(f"invalid kb name: {name!r}")

    def get_kb_lock(self, name: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._locks[name] = lock
            return lock
