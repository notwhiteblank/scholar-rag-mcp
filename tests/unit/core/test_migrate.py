import os
import sys
from pathlib import Path

import pytest

from scholar_rag.core.errors import ConfigError
from scholar_rag.core.migrate import migrate_legacy_data

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="linux-only migration")


def _legacy(home: Path) -> Path:
    return home / ".scholar-rag"


def test_no_legacy_is_noop(tmp_path):
    new_root = tmp_path / "xdg" / "scholar-rag"
    migrate_legacy_data(new_root, new_root / "qdrant-storage")
    assert not new_root.exists()


def test_legacy_data_moves_to_new_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = _legacy(tmp_path)
    (legacy / "kbs" / "demo").mkdir(parents=True)
    (legacy / "config.json").write_text("{}")
    new_root = tmp_path / "xdg" / "scholar-rag"
    migrate_legacy_data(new_root, new_root / "qdrant-storage")
    assert (new_root / "config.json").is_file()
    assert (new_root / "kbs" / "demo").is_dir()
    assert not legacy.exists()


def test_legacy_nested_storage_becomes_qdrant_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = _legacy(tmp_path)
    legacy.mkdir()
    (legacy / "jobs.sqlite3").write_bytes(b"x")
    storage = tmp_path / ".local" / "share" / "scholar-rag" / "qdrant"
    storage.mkdir(parents=True)
    (storage / "wall").write_bytes(b"y")
    new_root = tmp_path / ".local" / "share" / "scholar-rag"
    migrate_legacy_data(new_root, new_root / "qdrant-storage")
    assert (new_root / "qdrant-storage" / "wall").is_file()
    assert (new_root / "jobs.sqlite3").is_file()
    assert not legacy.exists()


def test_existing_new_root_with_content_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = _legacy(tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text("{}")
    new_root = tmp_path / "xdg" / "scholar-rag"
    new_root.mkdir(parents=True)
    (new_root / "kbs").mkdir()
    with pytest.warns(UserWarning):
        migrate_legacy_data(new_root, new_root / "qdrant-storage")
    assert legacy.is_dir()
    assert not (new_root / "config.json").exists()


def test_conflicting_entry_is_left_behind_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = _legacy(tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text('{"a": 1}')
    new_root = tmp_path / "xdg" / "scholar-rag"
    new_root.mkdir(parents=True)
    (new_root / "config.json").write_text('{"b": 2}')
    with pytest.warns(UserWarning):
        migrate_legacy_data(new_root, new_root / "qdrant-storage")
    assert (new_root / "config.json").read_text() == '{"b": 2}'
    assert (legacy / "config.json").is_file()


def test_move_failure_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = _legacy(tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text("{}")
    import scholar_rag.core.migrate as migrate_mod

    def boom(*args, **kwargs):
        raise OSError("disk on fire")

    monkeypatch.setattr(migrate_mod.shutil, "move", boom)
    new_root = tmp_path / "xdg" / "scholar-rag"
    with pytest.raises(ConfigError):
        migrate_legacy_data(new_root, new_root / "qdrant-storage")


def test_settings_load_migrates_legacy_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    legacy = _legacy(tmp_path)
    legacy.mkdir()
    (legacy / "config.json").write_text('{"CHAT_MODEL": "migrated-model"}')
    from scholar_rag.core.config import Settings

    settings = Settings.load()
    assert settings.chat_model == "migrated-model"
    assert settings.data_dir == tmp_path / "xdg" / "scholar-rag"
