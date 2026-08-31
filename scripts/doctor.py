#!/usr/bin/env python3
"""Environment doctor for scholar-rag-mcp.

Runs a series of quick checks (config, data dir, model services, MinerU,
Qdrant, vLLM ports) and prints a per-item pass/fail report. Exit code is 0
only when every mandatory check passes, 1 otherwise.

Usage:
    python scripts/doctor.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

try:
    from pydantic import ValidationError
    from scholar_rag.core.config import Settings
    from scholar_rag.core.errors import ConfigError
    from scholar_rag.models import get_chat_client, get_embedding_client, get_rerank_client
    from scholar_rag.store.qdrant_platform import QDRANT_VERSION, detect_asset
except ImportError:
    Settings = None  # type: ignore[assignment]
    ConfigError = Exception  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]
    get_chat_client = None  # type: ignore[assignment]
    get_embedding_client = None  # type: ignore[assignment]
    get_rerank_client = None  # type: ignore[assignment]
    QDRANT_VERSION = "1.12.5"  # type: ignore[assignment]

    def detect_asset(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("qdrant_platform unavailable")

_VLLM_PORTS = (8101, 8102, 8103)
_VLLM_ROLE = ("chat", "embed", "rerank")


class _Report:
    def __init__(self) -> None:
        self._items: list[tuple[bool, str, str]] = []

    def add(self, ok: bool, label: str, note: str) -> None:
        self._items.append((ok, label, note))

    def print(self) -> None:
        for ok, label, note in self._items:
            mark = "\u2713" if ok else "\u2717"
            print(f"{mark} {label}: {note}")

    def ok(self) -> bool:
        return all(ok for ok, _label, _note in self._items)


def _probe_write(directory: Path) -> str | None:
    try:
        with tempfile.TemporaryFile(dir=str(directory)) as handle:
            handle.write(b"probe")
    except OSError as exc:
        return str(exc)
    return None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _client_health(report: _Report, settings: Settings, group: str) -> None:
    backend = getattr(settings, f"{group}_backend")
    base_url = getattr(settings, f"{group}_base_url")
    model = getattr(settings, f"{group}_model")
    var = f"SCHOLAR_RAG_{group.upper()}_BASE_URL"
    if backend != "api":
        path = Path(model).expanduser()
        if not path.is_dir():
            report.add(False, f"{group} client ({backend})",
                       f"model path not found: {model!r}")
            return
        for module in ("torch", "transformers") if group == "chat" else ("torch",):
            if importlib.util.find_spec(module) is None:
                report.add(False, f"{group} client ({backend})",
                           f"dependency {module!r} not importable; use the "
                           "local-models pixi environment")
                return
        report.add(True, f"{group} client ({backend})",
                   f"model {model!r}; full load deferred to runtime")
        return
    if not base_url:
        report.add(False, f"{group} client ({backend})",
                   f"{var} is empty; set it to an OpenAI-compatible /v1 endpoint")
        return
    factory = {"chat": get_chat_client, "embed": get_embedding_client,
               "rerank": get_rerank_client}[group]
    try:
        healthy = factory().health()
    except Exception as exc:
        healthy = False
        note = f"{base_url} unreachable ({exc}); is the model server running?"
    else:
        note = f"{base_url} responded on /models"
    if not healthy:
        hint = ("scripts/serve_models.sh" if sys.platform == "linux"
                else "an OpenAI-compatible server (Ollama/LM Studio/llama.cpp); "
                     "scripts/serve_models.sh is Linux-only")
        note = f"{base_url} not healthy; check {var}, {group.upper()}_MODEL and {hint}"
    report.add(healthy, f"{group} client ({backend})", note)


def _mineru_report(report: _Report, settings: Settings) -> None:
    backend = settings.mineru_backend
    label = f"mineru ({backend})"
    if backend == "python":
        try:
            importlib.util.find_spec("mineru.cli.common")
            importable = True
        except (ImportError, ModuleNotFoundError):
            importable = False
        if not importable:
            report.add(False, label, "module 'mineru.cli.common' not importable; "
                      "run the doctor inside the 'mineru' pixi environment")
        else:
            report.add(True, label, "lazy import of mineru.cli.common succeeded")
        return
    if backend == "cli":
        binary = shutil.which(settings.mineru_bin)
        if binary is None:
            report.add(False, label, f"binary {settings.mineru_bin!r} not on PATH "
                       "(set SCHOLAR_RAG_MINERU_BIN)")
            return
        import subprocess
        completed = subprocess.run([binary, "--version"], capture_output=True,
                                   text=True, timeout=60, check=False)
        if completed.returncode == 0:
            version = (completed.stdout or completed.stderr).strip().splitlines()
            report.add(True, label, f"{binary} -> {version[0] if version else 'ok'}")
        else:
            report.add(False, label, f"{binary} returned {completed.returncode}")
        return
    api_url = settings.mineru_api_url
    import httpx
    try:
        response = httpx.get(f"{api_url}/", timeout=10)
        reachable = response.status_code < 500
    except httpx.HTTPError as exc:
        reachable = False
        note = f"{api_url} unreachable ({exc}); start mineru-api or fix SCHOLAR_RAG_MINERU_API_URL"
    else:
        note = f"{api_url} responded HTTP {response.status_code}"
    if not reachable:
        report.add(False, label, note)
    else:
        report.add(True, label, note)


def _qdrant_report(report: _Report, settings: Settings) -> None:
    if settings.qdrant_url:
        import httpx
        label = "qdrant (external)"
        try:
            response = httpx.get(f"{settings.qdrant_url}/readyz", timeout=5)
            ok = response.status_code == 200
        except httpx.HTTPError as exc:
            ok = False
            note = f"{settings.qdrant_url}/readyz unreachable ({exc})"
        else:
            note = f"{settings.qdrant_url} answered /readyz (HTTP {response.status_code})"
        if not ok:
            report.add(False, label, note)
        else:
            report.add(True, label, note)
        return
    label = "qdrant (auto-launch)"
    binary = settings.qdrant_bin or ""
    if binary:
        path = Path(binary).expanduser()
        if not path.is_file():
            report.add(False, label, f"SCHOLAR_RAG_QDRANT_BIN not found: {binary!r}")
            return
        report.add(True, label, f"binary found: {path}")
    else:
        try:
            asset = detect_asset()
        except Exception as exc:
            report.add(False, label, str(exc))
            return
        report.add(True, label, f"no SCHOLAR_RAG_QDRANT_BIN set; will look in "
                   f"{settings.data_dir / 'bin' / asset.binary_name} or auto-download "
                   f"{asset.asset_name} (v{QDRANT_VERSION})")
    storage = settings.qdrant_storage_dir
    try:
        storage.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.add(False, "qdrant storage dir", f"{storage} not creatable ({exc})")
        return
    error = _probe_write(storage)
    if error is not None:
        report.add(False, "qdrant storage dir", f"{storage} not writable ({error})")
        return
    report.add(True, "qdrant storage dir",
               f"{storage} writable; note: must reside on a local filesystem "
               "(9p/network mounts cause EINVAL)")


def _config_diagnosis(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    details: list[str] = []
    for error in errors()[:2]:
        loc = error.get("loc")
        field = loc[0] if isinstance(loc, tuple) and loc else None
        if isinstance(field, str) and field.isidentifier():
            label = f"SCHOLAR_RAG_{field.upper()}"
        else:
            label = str(field) if field is not None else "?"
        details.append(f"{label}: {error.get('msg', '')}")
    return "invalid configuration value; " + "; ".join(details)


def main(argv: list[str] | None = None) -> int:
    if Settings is None:
        print("\u2717 scholar_rag package: import failed - run this doctor inside the "
              "scholar-rag-mcp pixi environment (pixi run python scripts/doctor.py)")
        return 1
    report = _Report()
    try:
        settings = Settings.load()
    except (ConfigError, ValidationError) as exc:
        note = _config_diagnosis(exc) if isinstance(exc, ValidationError) else str(exc)
        report.add(False, "config", f"Settings.load() failed: {note}")
        try:
            settings = Settings()
        except (ConfigError, ValidationError) as exc2:
            note2 = _config_diagnosis(exc2) if isinstance(exc2, ValidationError) else str(exc2)
            report.add(False, "config", f"Settings() also failed: {note2}")
            report.print()
            print("SOME CHECKS FAILED (exit code 1)")
            return 1

    try:
        data_dir = settings.data_dir
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            report.add(False, "data dir", f"{data_dir} not creatable ({exc})")
        else:
            error = _probe_write(data_dir)
            if error is not None:
                report.add(False, "data dir", f"{data_dir} exists but not writable ({error})")
            else:
                report.add(True, "data dir", f"{data_dir} exists and is writable")
    except ValueError:
        report.add(False, "data dir", "DATA_DIR resolved to an invalid path")

    try:
        settings.validate_runtime()
        report.add(True, "validate_runtime", "no missing required *_BASE_URL for api backends")
    except ConfigError as exc:
        report.add(False, "validate_runtime", f"{exc}")

    for group in _VLLM_ROLE:
        _client_health(report, settings, group)

    _mineru_report(report, settings)
    _qdrant_report(report, settings)

    occupied = [port for port in _VLLM_PORTS if not _port_free(port)]
    if occupied:
        parts = ", ".join(
            f"{port}({role})" for port, role in zip(_VLLM_PORTS, _VLLM_ROLE) if port in occupied
        )
        report.add(True, "vllm ports", f"{parts} in use - assumed already running via "
                   "scripts/serve_models.sh (not an error)")
    else:
        if sys.platform == "linux":
            report.add(True, "vllm ports",
                       "8101/8102/8103 free on 127.0.0.1 (start them with scripts/serve_models.sh)")
        else:
            report.add(True, "vllm ports",
                       "8101/8102/8103 free on 127.0.0.1 (serve_models.sh is Linux-only; "
                       "point *_BASE_URL at any OpenAI-compatible server)")

    report.print()
    print()
    summary = "ALL CHECKS PASSED" if report.ok() else "SOME CHECKS FAILED (exit code 1)"
    print(summary)
    return 0 if report.ok() else 1


if __name__ == "__main__":
    sys.exit(main())