import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.store import qdrant_platform
from scholar_rag.store.qdrant_platform import (
    QdrantAsset,
    detect_asset,
    fetch_binary,
    release_url,
)


@pytest.mark.parametrize(
    "platform_name,machine,asset_name,kind,binary",
    [
        ("linux", "x86_64", "qdrant-x86_64-unknown-linux-gnu.tar.gz", "tar.gz", "qdrant"),
        ("darwin", "x86_64", "qdrant-x86_64-apple-darwin.tar.gz", "tar.gz", "qdrant"),
        ("darwin", "arm64", "qdrant-aarch64-apple-darwin.tar.gz", "tar.gz", "qdrant"),
        ("win32", "AMD64", "qdrant-x86_64-pc-windows-msvc.zip", "zip", "qdrant.exe"),
    ],
)
def test_detect_asset_supported(platform_name, machine, asset_name, kind, binary):
    asset = detect_asset(platform_name, machine)
    assert asset.asset_name == asset_name
    assert asset.archive_kind == kind
    assert asset.binary_name == binary


@pytest.mark.parametrize(
    "machine,resolved",
    [
        ("x86_64", "x86_64"),
        ("AMD64", "x86_64"),
        ("x64", "x86_64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("ARM64", "arm64"),
    ],
)
def test_machine_normalization(machine, resolved):
    if resolved == "x86_64":
        assert detect_asset("linux", machine).asset_name == "qdrant-x86_64-unknown-linux-gnu.tar.gz"
    else:
        assert detect_asset("darwin", machine).asset_name == "qdrant-aarch64-apple-darwin.tar.gz"


@pytest.mark.parametrize(
    "platform_name,machine",
    [("linux", "aarch64"), ("freebsd", "x86_64"), ("win32", "arm64")],
)
def test_detect_asset_unsupported_raises(platform_name, machine):
    with pytest.raises(ServiceUnavailableError) as excinfo:
        detect_asset(platform_name, machine)
    assert "SCHOLAR_RAG_QDRANT_URL" in str(excinfo.value)


def test_release_url_shape():
    asset = detect_asset("linux", "x86_64")
    url = release_url(asset)
    assert url == (
        "https://github.com/qdrant/qdrant/releases/download/"
        f"v{qdrant_platform.QDRANT_VERSION}/{asset.asset_name}"
    )


def _make_tar_gz(tmp_path: Path, arcname: str) -> Path:
    payload = tmp_path / "payload"
    payload.write_bytes(b"fake-binary")
    archive = tmp_path / "fake.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname=arcname)
    payload.unlink()
    return archive


def _make_zip(tmp_path: Path, arcname: str) -> Path:
    archive = tmp_path / "fake.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(arcname, b"fake-binary")
    return archive


def _stub_download(monkeypatch, archive_path: Path):
    def fake_download(url: str, dest: Path) -> None:
        import shutil

        shutil.copyfile(archive_path, dest)

    monkeypatch.setattr(qdrant_platform, "_download", fake_download)


def test_fetch_binary_tar_gz_nested_layout(tmp_path, monkeypatch):
    archive = _make_tar_gz(tmp_path, "qdrant-1.12.5/qdrant")
    _stub_download(monkeypatch, archive)
    cache = tmp_path / "cache"
    asset = QdrantAsset("fake.tar.gz", "tar.gz", "qdrant")
    result = fetch_binary(cache, asset)
    assert result == cache / "qdrant"
    assert result.read_bytes() == b"fake-binary"
    if os.name == "posix":
        assert os.access(result, os.X_OK)


def test_fetch_binary_zip_root_layout(tmp_path, monkeypatch):
    archive = _make_zip(tmp_path, "qdrant.exe")
    _stub_download(monkeypatch, archive)
    cache = tmp_path / "cache"
    asset = QdrantAsset("fake.zip", "zip", "qdrant.exe")
    result = fetch_binary(cache, asset)
    assert result == cache / "qdrant.exe"
    assert result.read_bytes() == b"fake-binary"


def test_fetch_binary_cache_hit_skips_download(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "qdrant").write_bytes(b"cached")

    def boom(url: str, dest: Path) -> None:
        raise AssertionError("download must not happen on cache hit")

    monkeypatch.setattr(qdrant_platform, "_download", boom)
    asset = QdrantAsset("fake.tar.gz", "tar.gz", "qdrant")
    assert fetch_binary(cache, asset) == cache / "qdrant"


def test_fetch_binary_zip_path_traversal_rejected(tmp_path, monkeypatch):
    archive = _make_zip(tmp_path, "../evil.txt")
    _stub_download(monkeypatch, archive)
    asset = QdrantAsset("fake.zip", "zip", "qdrant")
    with pytest.raises(ServiceUnavailableError):
        fetch_binary(tmp_path / "cache", asset)


def test_fetch_binary_missing_binary_in_archive(tmp_path, monkeypatch):
    archive = _make_zip(tmp_path, "something-else.exe")
    _stub_download(monkeypatch, archive)
    asset = QdrantAsset("fake.zip", "zip", "qdrant.exe")
    with pytest.raises(ServiceUnavailableError):
        fetch_binary(tmp_path / "cache", asset)
