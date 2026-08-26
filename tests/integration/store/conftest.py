import os
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPIKE_BINARY = _REPO_ROOT / "spike" / "out" / "qdrant" / "qdrant"
_DOWNLOAD_URL = (
    "https://github.com/qdrant/qdrant/releases/download/"
    "v1.12.5/qdrant-x86_64-unknown-linux-gnu.tar.gz"
)
_READY_TIMEOUT = 60.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _resolve_binary(cache_dir: Path) -> Path:
    configured = os.environ.get("SCHOLAR_RAG_QDRANT_BIN")
    if configured and Path(configured).is_file():
        return Path(configured)
    if _SPIKE_BINARY.is_file():
        return _SPIKE_BINARY
    cached = cache_dir / "qdrant"
    if cached.is_file():
        return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "qdrant.tar.gz"
    urllib.request.urlretrieve(_DOWNLOAD_URL, archive)
    with tarfile.open(archive, "r:gz") as handle:
        member = next(m for m in handle.getmembers() if Path(m.name).name == "qdrant")
        handle.extract(member, cache_dir)
        extracted = cache_dir / Path(member.name).name
        extracted.chmod(0o755)
    return extracted


def _wait_ready(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"qdrant exited early with code {process.returncode}")
        try:
            response = httpx.get(f"{url}/readyz", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"qdrant not ready on {url} within {_READY_TIMEOUT}s")


@pytest.fixture(scope="session")
def qdrant_instance(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("qdrant-instance")
    binary = _resolve_binary(root / "bin-cache")
    bin_path = root / "qdrant"
    shutil.copy2(binary, bin_path)
    port = _free_port()
    storage = root / "storage"
    config = root / "qdrant.yaml"
    config.write_text(
        f"storage:\n  storage_path: {storage}\nservice:\n  http_port: {port}\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [str(bin_path), "--config-path", str(config), "--disable-telemetry"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(root),
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_ready(url, process)
    except Exception:
        process.kill()
        raise
    yield {"url": url, "binary": bin_path, "storage": storage}
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path, qdrant_instance):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_URL", qdrant_instance["url"])
    from scholar_rag.store.qdrant_manager import reset_qdrant_manager

    reset_qdrant_manager()
    yield
    reset_qdrant_manager()
