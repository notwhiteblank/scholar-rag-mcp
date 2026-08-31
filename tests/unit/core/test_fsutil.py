import os
import sys

import pytest

from scholar_rag.core.fsutil import move_dir


def test_move_dir_moves_directory(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "file.txt").write_text("x")
    target = tmp_path / "dst"
    move_dir(source, target)
    assert (target / "file.txt").read_text() == "x"
    assert not source.exists()


def test_move_dir_retries_on_permission_error(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    target = tmp_path / "dst"
    calls = {"n": 0}
    real_replace = os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3 and sys.platform == "win32":
            raise PermissionError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    move_dir(source, target)
    assert target.is_dir()
    assert calls["n"] == (3 if sys.platform == "win32" else 1)


def test_move_dir_raises_after_exhausting_retries(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("retry path only exercised on win32")
    source = tmp_path / "src"
    source.mkdir()

    def deny(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", deny)
    monkeypatch.setattr("scholar_rag.core.fsutil._RETRY_DELAY_SECONDS", 0.001)
    with pytest.raises(PermissionError):
        move_dir(source, tmp_path / "dst")
