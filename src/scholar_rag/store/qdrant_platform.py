from __future__ import annotations

import os
import platform
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from scholar_rag.core.errors import ServiceUnavailableError

QDRANT_VERSION = "1.12.5"
_RELEASE_URL_TEMPLATE = "https://github.com/qdrant/qdrant/releases/download/v{version}/{asset}"


@dataclass(frozen=True)
class QdrantAsset:
    asset_name: str
    archive_kind: str
    binary_name: str


_ASSETS: dict[tuple[str, str], QdrantAsset] = {
    ("linux", "x86_64"): QdrantAsset(
        "qdrant-x86_64-unknown-linux-gnu.tar.gz", "tar.gz", "qdrant"
    ),
    ("darwin", "x86_64"): QdrantAsset(
        "qdrant-x86_64-apple-darwin.tar.gz", "tar.gz", "qdrant"
    ),
    ("darwin", "arm64"): QdrantAsset(
        "qdrant-aarch64-apple-darwin.tar.gz", "tar.gz", "qdrant"
    ),
    ("win32", "x86_64"): QdrantAsset(
        "qdrant-x86_64-pc-windows-msvc.zip", "zip", "qdrant.exe"
    ),
}


def _normalize_machine(machine: str) -> str:
    lowered = machine.strip().lower()
    if lowered in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if lowered in ("arm64", "aarch64"):
        return "arm64"
    return lowered


def detect_asset(platform_name: str | None = None, machine: str | None = None) -> QdrantAsset:
    plat = platform_name if platform_name is not None else sys.platform
    mach = _normalize_machine(machine if machine is not None else platform.machine())
    asset = _ASSETS.get((plat, mach))
    if asset is None:
        raise ServiceUnavailableError(
            f"no bundled Qdrant binary for platform {plat}/{mach}; "
            "set SCHOLAR_RAG_QDRANT_URL to an external instance "
            "or SCHOLAR_RAG_QDRANT_BIN to a local binary"
        )
    return asset


def release_url(asset: QdrantAsset) -> str:
    return _RELEASE_URL_TEMPLATE.format(version=QDRANT_VERSION, asset=asset.asset_name)


def _download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def _extract(archive: Path, kind: str, dest_dir: Path) -> None:
    if kind == "zip":
        root = dest_dir.resolve()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if not (root / member).resolve().is_relative_to(root):
                    raise OSError(f"unsafe path in qdrant archive: {member}")
            handle.extractall(dest_dir)
        return
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(dest_dir, filter="data")


def fetch_binary(cache_dir: Path, asset: QdrantAsset | None = None) -> Path:
    current = asset or detect_asset()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / current.binary_name
    if cached.is_file():
        return cached
    tmp_dir = Path(tempfile.mkdtemp(prefix="qdrant-download-", dir=str(cache_dir)))
    try:
        archive = tmp_dir / f"qdrant.{current.archive_kind}"
        url = release_url(current)
        try:
            _download(url, archive)
            _extract(archive, current.archive_kind, tmp_dir)
        except (httpx.HTTPError, tarfile.TarError, zipfile.BadZipFile, OSError) as exc:
            raise ServiceUnavailableError(
                f"failed to download qdrant v{QDRANT_VERSION} from {url}: {exc}"
            ) from exc
        extracted = next(
            (path for path in tmp_dir.rglob(current.binary_name) if path.is_file()), None
        )
        if extracted is None:
            raise ServiceUnavailableError(
                f"qdrant archive {current.asset_name} does not contain {current.binary_name}"
            )
        if os.name == "posix":
            extracted.chmod(0o755)
        shutil.move(str(extracted), cached)
        return cached
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
