import json
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

_DIM = 8


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
    root = tmp_path_factory.mktemp("qdrant-server-instance")
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


class _ModelHandler(BaseHTTPRequestHandler):
    def log_message(self, msg, *args):  # noqa: A002
        return None

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"data": [{"id": "fake-model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        if self.path == "/embeddings":
            inputs = request.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            vectors = [[0.5] * _DIM + [float(index + 1) * 0.01] for index in range(len(inputs))]
            self._reply(
                {
                    "data": [
                        {"index": index, "embedding": vector}
                        for index, vector in enumerate(vectors)
                    ]
                }
            )
            return
        if self.path == "/rerank":
            documents = request.get("documents", [])
            self._reply(
                {
                    "results": [
                        {"index": index, "relevance_score": 0.5 - index * 0.01}
                        for index in range(len(documents))
                    ]
                }
            )
            return
        self._reply({})


@pytest.fixture
def fake_model_url() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
