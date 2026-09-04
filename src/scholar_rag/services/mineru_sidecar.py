from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import PipelineStageError

_STARTUP_TIMEOUT = 600.0
_POLL_INTERVAL = 2.0
_HEALTH_TIMEOUT = 2.0
_STOP_GRACE = 5.0
_MANAGED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class SidecarError(PipelineStageError):
    pass


def managed_env_dir(settings: Settings) -> Path:
    return settings.data_dir / "mineru-env"


def managed_api_binary(settings: Settings) -> Path:
    env_dir = managed_env_dir(settings)
    if sys.platform == "win32":
        return env_dir / "Scripts" / "mineru-api.exe"
    return env_dir / "bin" / "mineru-api"


def _pid_file(settings: Settings) -> Path:
    return settings.data_dir / "mineru-api.pid"


def _log_file(settings: Settings) -> Path:
    return settings.data_dir / "mineru-api.log"


def _parse_host_port(api_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(api_url)
    if parsed.hostname is None or parsed.port is None:
        raise SidecarError(f"cannot parse host/port from mineru_api_url: {api_url!r}")
    return parsed.hostname, parsed.port


def is_managed_local(settings: Settings) -> bool:
    if not settings.mineru_managed or settings.mineru_backend != "api":
        return False
    host, _port = _parse_host_port(settings.mineru_api_url)
    return host in _MANAGED_HOSTS


def is_reachable(api_url: str, timeout: float = _HEALTH_TIMEOUT) -> bool:
    try:
        httpx.get(f"{api_url}/health", timeout=timeout)
    except httpx.HTTPError:
        return False
    return True


def _spawn(settings: Settings) -> subprocess.Popen[bytes]:
    binary = managed_api_binary(settings)
    if not binary.is_file():
        raise SidecarError(
            f"managed MinerU not found at {binary}; "
            "run: scholar-rag-mcp install --with-mineru"
        )
    host, port = _parse_host_port(settings.mineru_api_url)
    log_path = _log_file(settings)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(binary), "--host", host, "--port", str(port)]
    with log_path.open("ab") as log_handle:
        if sys.platform == "win32":
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
    return process


def ensure_running(settings: Settings) -> None:
    if not is_managed_local(settings):
        return
    if is_reachable(settings.mineru_api_url):
        return
    process = _spawn(settings)
    _pid_file(settings).write_text(str(process.pid), encoding="utf-8")
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if is_reachable(settings.mineru_api_url):
            return
        if process.poll() is not None:
            raise SidecarError(
                f"mineru-api exited during startup (code {process.poll()}); "
                f"see {_log_file(settings)}"
            )
        time.sleep(_POLL_INTERVAL)
    raise SidecarError(
        f"mineru-api did not become ready within {_STARTUP_TIMEOUT:.0f}s; "
        f"first start downloads models; see {_log_file(settings)}"
    )


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in completed.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_sidecar(settings: Settings) -> str:
    pid_path = _pid_file(settings)
    if not pid_path.is_file():
        return "absent"
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        pid_path.unlink(missing_ok=True)
        return "absent"
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pid_path.unlink(missing_ok=True)
        return "absent"
    deadline = time.monotonic() + _STOP_GRACE
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.2)
    if _alive(pid):
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)
    return "stopped"
