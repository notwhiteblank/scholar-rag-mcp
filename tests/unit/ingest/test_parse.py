from __future__ import annotations

import os
from pathlib import Path

import pytest

import scholar_rag.services.mineru as mineru_mod
from scholar_rag.core.errors import PipelineStageError
from scholar_rag.ingest.parse import stage_parse
from scholar_rag.ingest.staging import stage_pdf


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def staged(tmp_path: Path):
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return stage_pdf(source, tmp_path / "workdir")


def test_cache_hit_skips_parse(staged, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{staged.content_hash}.md").write_text("cached markdown", encoding="utf-8")
    calls: list[Path] = []

    def fake_parse(pdf_path: Path) -> str:
        calls.append(pdf_path)
        return "should not be reached"

    monkeypatch.setattr(mineru_mod, "parse", fake_parse)
    markdown = stage_parse(staged, cache)
    assert markdown == "cached markdown"
    assert calls == []


@pytest.mark.parametrize(
    "backend, attr, arity",
    [
        ("python", "_parse_via_python", 1),
        ("cli", "_parse_via_cli", 2),
        ("api", "_parse_via_api", 2),
    ],
)
def test_parse_dispatches_to_configured_backend(
    backend: str, attr: str, arity: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setenv("SCHOLAR_RAG_MINERU_BACKEND", backend)

    def fake_inner(*args: object) -> str:
        assert len(args) == arity
        pdf_path = Path(str(args[0]))
        return f"parsed {pdf_path.name}"

    monkeypatch.setattr(mineru_mod, attr, fake_inner)
    assert mineru_mod.parse(source) == "parsed doc.pdf"


def test_parse_failure_raises_with_backend_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setenv("SCHOLAR_RAG_MINERU_BACKEND", "python")

    def boom(pdf_path: Path) -> str:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(mineru_mod, "_parse_via_python", boom)
    with pytest.raises(PipelineStageError) as excinfo:
        mineru_mod.parse(source)
    message = str(excinfo.value)
    assert "python" in message
    assert "kaboom" in message


def test_cache_miss_calls_parse_and_writes_cache(
    staged, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    calls: list[Path] = []

    def fake_parse(pdf_path: Path) -> str:
        calls.append(pdf_path)
        return "fresh markdown"

    monkeypatch.setattr(mineru_mod, "parse", fake_parse)
    markdown = stage_parse(staged, cache)
    assert markdown == "fresh markdown"
    assert calls == [staged.path]
    cached = cache / f"{staged.content_hash}.md"
    assert cached.read_text(encoding="utf-8") == "fresh markdown"


def test_parse_failure_propagates_and_does_not_cache(
    staged, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"

    def fake_parse(pdf_path: Path) -> str:
        raise PipelineStageError("mineru failed")

    monkeypatch.setattr(mineru_mod, "parse", fake_parse)
    with pytest.raises(PipelineStageError):
        stage_parse(staged, cache)
    assert not (cache / f"{staged.content_hash}.md").exists()
