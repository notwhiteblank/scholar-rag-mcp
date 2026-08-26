from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import scholar_rag.services.pdf2doi as pdf2doi_mod


def _fake_fitz(first_text: str) -> SimpleNamespace:
    page = SimpleNamespace(get_text=lambda: first_text)
    doc = MagicMock()
    doc.__len__ = lambda self: 1  # type: ignore[no-redef]
    doc.__getitem__ = lambda self, i: page  # type: ignore[no-redef]
    return SimpleNamespace(open=lambda path: doc)


def _fake_pdf2doi(identifier: str) -> SimpleNamespace:
    module = SimpleNamespace()
    module.config = SimpleNamespace(set=lambda *args, **kwargs: None)
    module.pdf2doi = lambda path: {"identifier": identifier, "method": "fake"}
    return module


def _regex_only(monkeypatch: pytest.MonkeyPatch, first_text: str) -> None:
    def raiser() -> None:
        raise ImportError("pdf2doi not installed")

    monkeypatch.setattr(pdf2doi_mod, "_import_pdf2doi", raiser)
    monkeypatch.setattr(pdf2doi_mod, "_import_fitz", lambda: _fake_fitz(first_text))


def test_regex_extraction_from_first_page(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _regex_only(monkeypatch, "Some text\nDOI: 10.1000/xyz.12345. Here is more.")
    assert pdf2doi_mod.extract_doi(pdf) == "10.1000/xyz.12345"


def test_regex_component_doi_is_stripped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _regex_only(monkeypatch, "elife article\n10.7554/eLife.06807.001\n")
    doi, method = pdf2doi_mod.extract_doi_with_info(pdf)
    assert doi == "10.7554/eLife.06807"
    assert "component_fix" in method


def test_regex_junk_cleaned(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _regex_only(monkeypatch, "10.1000/xyz.12345.\n")
    assert pdf2doi_mod.extract_doi(pdf) == "10.1000/xyz.12345"


def test_no_doi_returns_none(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _regex_only(monkeypatch, "The quick brown fox jumps over the lazy dog.\n")
    assert pdf2doi_mod.extract_doi(pdf) is None


def test_pdf2doi_library_preferred_over_regex(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(pdf2doi_mod, "_import_pdf2doi", lambda: _fake_pdf2doi("10.1000/lib.00001"))
    monkeypatch.setattr(pdf2doi_mod, "_import_fitz", lambda: _fake_fitz("10.1000/regex.00002"))
    doi, method = pdf2doi_mod.extract_doi_with_info(pdf)
    assert doi == "10.1000/lib.00001"
    assert method.startswith("pdf2doi")


def test_pdf2doi_exception_falls_back_to_regex(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    module = SimpleNamespace(
        config=SimpleNamespace(set=lambda *args, **kwargs: None),
        pdf2doi=lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(pdf2doi_mod, "_import_pdf2doi", lambda: module)
    monkeypatch.setattr(pdf2doi_mod, "_import_fitz", lambda: _fake_fitz("10.1000/regex.00003"))
    doi, method = pdf2doi_mod.extract_doi_with_info(pdf)
    assert doi == "10.1000/regex.00003"
    assert method == "regex_first_page"


def test_text_extraction_unavailable_returns_empty(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def raiser() -> None:
        raise ImportError("fitz not installed")

    monkeypatch.setattr(pdf2doi_mod, "_import_fitz", raiser)
    assert pdf2doi_mod._extract_text_first_pages(pdf) == ""


def test_line_break_in_doi_is_rejoined(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _regex_only(monkeypatch, "See doi:\n10.1000/joined.\n12345 for details\n")
    assert pdf2doi_mod.extract_doi(pdf) == "10.1000/joined.12345"
