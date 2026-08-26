from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scholar_rag.core.errors import PipelineStageError
from scholar_rag.ingest.staging import stage_pdf

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


def _pdf_fixture(name: str) -> Path:
    path = FIXTURES_DIR / f"{name}.pdf"
    assert path.is_file(), f"missing fixture {path}; run tests/fixtures/gen_pdfs.py"
    return path


def test_stage_pdf_copies_binary_and_computes_hash(tmp_path: Path) -> None:
    source = _pdf_fixture("paper_a")
    work = tmp_path / "work"
    staged = stage_pdf(source, work)

    assert staged.source_path == source
    assert staged.path == work / "paper_a.pdf"
    assert staged.name == "paper_a"
    assert staged.path.read_bytes() == source.read_bytes()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert staged.content_hash == expected
    assert len(staged.content_hash) == 64


def test_stage_pdf_rejects_bad_extension(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(PipelineStageError):
        stage_pdf(source, tmp_path / "work")


def test_stage_pdf_rejects_missing_pdf_magic(tmp_path: Path) -> None:
    source = tmp_path / "fake.pdf"
    source.write_bytes(b"this is not a pdf\n")
    with pytest.raises(PipelineStageError):
        stage_pdf(source, tmp_path / "work")


def test_stage_pdf_rejects_truncated_magic(tmp_path: Path) -> None:
    source = tmp_path / "truncated.pdf"
    source.write_bytes(b"%PD")
    with pytest.raises(PipelineStageError):
        stage_pdf(source, tmp_path / "work")


def test_stage_pdf_creates_work_dir(tmp_path: Path) -> None:
    source = _pdf_fixture("paper_b")
    work = tmp_path / "nested" / "work"
    stage_pdf(source, work)
    assert work.is_dir()
    assert (work / "paper_b.pdf").is_file()


def test_stage_pdf_does_not_copy_to_self(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    source = work / "already_staged.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    staged = stage_pdf(source, work)
    assert staged.path == source
    assert staged.path.read_bytes() == b"%PDF-1.4\n%%EOF\n"
