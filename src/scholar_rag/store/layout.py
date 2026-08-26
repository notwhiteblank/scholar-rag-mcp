from __future__ import annotations

from pathlib import Path

from scholar_rag.core.config import Settings


def _current_settings() -> Settings:
    return Settings.load()


def kbs_root(settings: Settings | None = None) -> Path:
    current = settings or _current_settings()
    return current.data_dir / "kbs"


def kb_dir(name: str, settings: Settings | None = None) -> Path:
    return kbs_root(settings) / name


def catalog_path(name: str, settings: Settings | None = None) -> Path:
    return kb_dir(name, settings) / "catalog.sqlite3"


def documents_dir(name: str, settings: Settings | None = None) -> Path:
    return kb_dir(name, settings) / "documents"


def parse_cache_dir(settings: Settings | None = None) -> Path:
    current = settings or _current_settings()
    return current.data_dir / "cache" / "parse"


def jobs_db_path(settings: Settings | None = None) -> Path:
    current = settings or _current_settings()
    return current.data_dir / "jobs.sqlite3"


def bin_dir(settings: Settings | None = None) -> Path:
    current = settings or _current_settings()
    return current.data_dir / "bin"


def ensure_data_dir(settings: Settings | None = None) -> None:
    current = settings or _current_settings()
    for path in (
        current.data_dir,
        kbs_root(current),
        parse_cache_dir(current),
        bin_dir(current),
        current.qdrant_storage_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
