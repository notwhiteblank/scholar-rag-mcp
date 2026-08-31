from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
from qdrant_client import QdrantClient

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.store.layout import bin_dir
from scholar_rag.store.qdrant_platform import QDRANT_VERSION, detect_asset, fetch_binary

_READY_TIMEOUT_SECONDS = 30.0
_READY_INTERVAL_SECONDS = 0.5


class QdrantManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings.load()
        self._client: QdrantClient | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._runtime_config: Path | None = None
        atexit.register(self.stop)

    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self) -> QdrantClient:
        settings = self._settings
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url)
        binary = self._resolve_binary()
        process = self._process
        if process is None or process.poll() is not None:
            self._start(binary)
        return QdrantClient(url=f"http://127.0.0.1:{settings.qdrant_port}")

    def _resolve_binary(self) -> Path:
        settings = self._settings
        if settings.qdrant_bin:
            path = Path(settings.qdrant_bin).expanduser()
            if not path.is_file():
                raise ServiceUnavailableError(f"QDRANT_BIN not found: {settings.qdrant_bin}")
            return path
        target = bin_dir(settings) / detect_asset().binary_name
        if target.is_file():
            return target
        return self._download_binary(target)

    def _download_binary(self, target: Path) -> Path:
        return fetch_binary(target.parent)

    def _start(self, binary: Path) -> None:
        settings = self._settings
        storage = settings.qdrant_storage_dir
        storage.mkdir(parents=True, exist_ok=True)
        config_path = self._write_runtime_config(storage, settings.qdrant_port)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        self._process = subprocess.Popen(
            [str(binary), "--config-path", str(config_path), "--disable-telemetry"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(storage),
            creationflags=creationflags,
        )
        url = f"http://127.0.0.1:{settings.qdrant_port}"
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            try:
                response = httpx.get(f"{url}/readyz", timeout=_READY_INTERVAL_SECONDS)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(_READY_INTERVAL_SECONDS)
        self.stop()
        raise ServiceUnavailableError(
            f"qdrant v{QDRANT_VERSION} failed to become ready on {url}"
            f" within {_READY_TIMEOUT_SECONDS}s"
        )

    def _write_runtime_config(self, storage: Path, port: int) -> Path:
        descriptor, path = tempfile.mkstemp(prefix="qdrant-", suffix=".yaml")
        quoted_storage = str(storage).replace("\\", "\\\\").replace('"', '\\"')
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                'storage:\n'
                f'  storage_path: "{quoted_storage}"\n'
                'service:\n'
                f'  http_port: {port}\n'
            )
        self._runtime_config = Path(path)
        return self._runtime_config

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        config = self._runtime_config
        self._runtime_config = None
        if config is not None:
            with contextlib.suppress(OSError):
                config.unlink(missing_ok=True)


_manager_lock = threading.Lock()
_manager: QdrantManager | None = None


def get_qdrant_manager() -> QdrantManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = QdrantManager()
        return _manager


def reset_qdrant_manager() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop()
            _manager = None
