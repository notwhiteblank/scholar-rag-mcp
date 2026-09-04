from __future__ import annotations

from pathlib import Path

import pytest

from scholar_rag.core.config import Settings
from scholar_rag.services import mineru_sidecar
from scholar_rag.services.mineru_sidecar import (
    SidecarError,
    ensure_running,
    is_managed_local,
    is_reachable,
    managed_api_binary,
    stop_sidecar,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "data_dir": tmp_path,
        "mineru_backend": "api",
        "mineru_managed": True,
        "mineru_api_url": "http://127.0.0.1:8010",
    }
    base.update(overrides)
    return Settings(**base)


def test_managed_api_binary_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(mineru_sidecar.sys, "platform", "linux")
    binary = managed_api_binary(_settings(tmp_path))
    assert binary == tmp_path / "mineru-env" / "bin" / "mineru-api"


def test_managed_api_binary_win32(tmp_path, monkeypatch):
    monkeypatch.setattr(mineru_sidecar.sys, "platform", "win32")
    binary = managed_api_binary(_settings(tmp_path))
    assert binary == tmp_path / "mineru-env" / "Scripts" / "mineru-api.exe"


def test_is_managed_local_requires_flag(tmp_path):
    assert is_managed_local(_settings(tmp_path, mineru_managed=False)) is False


def test_is_managed_local_rejects_remote_host(tmp_path):
    settings = _settings(tmp_path, mineru_api_url="http://10.0.0.5:8010")
    assert is_managed_local(settings) is False


def test_is_managed_local_accepts_localhost(tmp_path):
    settings = _settings(tmp_path, mineru_api_url="http://localhost:8010")
    assert is_managed_local(settings) is True


def test_is_reachable_true_on_any_http_response(monkeypatch):
    class _Resp:
        status_code = 503

    monkeypatch.setattr(
        mineru_sidecar.httpx, "get", lambda url, timeout: _Resp()
    )
    assert is_reachable("http://127.0.0.1:8010") is True


def test_is_reachable_false_on_http_error(monkeypatch):
    import httpx

    def _boom(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mineru_sidecar.httpx, "get", _boom)
    assert is_reachable("http://127.0.0.1:8010") is False


def test_ensure_running_noop_when_not_managed(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        mineru_sidecar, "is_reachable", lambda url, timeout=2.0: calls.append(url) or False
    )
    ensure_running(_settings(tmp_path, mineru_managed=False))
    assert calls == []


def test_ensure_running_noop_when_already_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(mineru_sidecar, "is_reachable", lambda url, timeout=2.0: True)
    spawned: list[object] = []
    monkeypatch.setattr(mineru_sidecar, "_spawn", lambda settings: spawned.append(1))
    ensure_running(_settings(tmp_path))
    assert spawned == []


def test_ensure_running_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(mineru_sidecar, "is_reachable", lambda url, timeout=2.0: False)
    with pytest.raises(SidecarError, match="install --with-mineru"):
        ensure_running(_settings(tmp_path))


class _FakeProc:
    def __init__(self, exit_code=None):
        self._exit_code = exit_code
        self.pid = 4242

    def poll(self):
        return self._exit_code


def test_ensure_running_spawns_and_waits_for_health(tmp_path, monkeypatch):
    binary = tmp_path / "mineru-env" / "bin" / "mineru-api"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    states = {"reachable": False, "spawns": 0}

    def _fake_reachable(url, timeout=2.0):
        return states["reachable"]

    def _fake_spawn(settings):
        states["spawns"] += 1
        states["reachable"] = True
        return _FakeProc()

    monkeypatch.setattr(mineru_sidecar, "is_reachable", _fake_reachable)
    monkeypatch.setattr(mineru_sidecar, "_spawn", _fake_spawn)
    monkeypatch.setattr(mineru_sidecar.time, "sleep", lambda seconds: None)
    ensure_running(_settings(tmp_path))
    assert states["spawns"] == 1
    pid_text = (tmp_path / "mineru-api.pid").read_text(encoding="utf-8")
    assert pid_text == "4242"


def test_ensure_running_spawn_exit_raises_with_log(tmp_path, monkeypatch):
    binary = tmp_path / "mineru-env" / "bin" / "mineru-api"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    monkeypatch.setattr(mineru_sidecar, "is_reachable", lambda url, timeout=2.0: False)
    monkeypatch.setattr(mineru_sidecar, "_spawn", lambda settings: _FakeProc(exit_code=3))
    monkeypatch.setattr(mineru_sidecar.time, "sleep", lambda seconds: None)
    with pytest.raises(SidecarError, match="exited during startup"):
        ensure_running(_settings(tmp_path))


def test_stop_sidecar_absent_without_pid_file(tmp_path):
    assert stop_sidecar(_settings(tmp_path)) == "absent"


def test_stop_sidecar_kills_pid(tmp_path, monkeypatch):
    pid_file = tmp_path / "mineru-api.pid"
    pid_file.write_text("4242", encoding="utf-8")
    killed: list[int] = []
    monkeypatch.setattr(mineru_sidecar, "_alive", lambda pid: False)
    monkeypatch.setattr(mineru_sidecar.os, "kill", lambda pid, sig: killed.append(pid))
    assert stop_sidecar(_settings(tmp_path)) == "stopped"
    assert killed == [4242]
    assert not pid_file.exists()


def test_stop_sidecar_invalid_pid_file(tmp_path):
    (tmp_path / "mineru-api.pid").write_text("not-a-pid", encoding="utf-8")
    assert stop_sidecar(_settings(tmp_path)) == "absent"
    assert not (tmp_path / "mineru-api.pid").exists()
