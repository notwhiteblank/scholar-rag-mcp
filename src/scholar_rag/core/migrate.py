from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

from scholar_rag.core.errors import ConfigError

_LEGACY_DATA_DIR = "~/.scholar-rag"
_LEGACY_QDRANT_STORAGE = "~/.local/share/scholar-rag/qdrant"
_LEGACY_STORAGE_NAME = "qdrant"


def migrate_legacy_data(new_data_dir: Path, new_storage_dir: Path) -> None:
    if sys.platform != "linux":
        return
    legacy_data = Path(_LEGACY_DATA_DIR).expanduser()
    if not legacy_data.is_dir():
        return
    legacy_storage = Path(_LEGACY_QDRANT_STORAGE).expanduser()
    storage_is_nested = legacy_storage.is_dir() and legacy_storage.parent == new_data_dir
    if new_data_dir.exists():
        entries = [entry.name for entry in new_data_dir.iterdir()]
        allowed = {_LEGACY_STORAGE_NAME} if storage_is_nested else set()
        if any(name not in allowed for name in entries):
            warnings.warn(
                f"legacy data directory {legacy_data} not migrated: "
                f"{new_data_dir} already exists with data; "
                "move it manually or set SCHOLAR_RAG_DATA_DIR",
                stacklevel=2,
            )
            return
    try:
        new_data_dir.mkdir(parents=True, exist_ok=True)
        if legacy_storage.is_dir() and not new_storage_dir.exists():
            shutil.move(str(legacy_storage), str(new_storage_dir))
        leftover: list[str] = []
        for entry in sorted(legacy_data.iterdir()):
            target = new_data_dir / entry.name
            if target.exists():
                leftover.append(entry.name)
                continue
            shutil.move(str(entry), str(target))
        if leftover:
            warnings.warn(
                f"migration left conflicting entries in {legacy_data}: {', '.join(leftover)}",
                stacklevel=2,
            )
        try:
            legacy_data.rmdir()
        except OSError:
            warnings.warn(
                f"legacy directory {legacy_data} not empty after migration",
                stacklevel=2,
            )
    except (OSError, shutil.Error) as exc:
        raise ConfigError(
            f"failed to migrate legacy data {legacy_data} to {new_data_dir}: {exc}; "
            "move your data manually and set SCHOLAR_RAG_DATA_DIR"
        ) from exc
