from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pytest
import respx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from scholar_rag.core.types import MetadataResult  # noqa: E402
from scholar_rag.ingest import metadata as metadata_mod  # noqa: E402
from scholar_rag.ingest.staging import StagedPdf, stage_pdf  # noqa: E402

CROSSREF_WORKS = "https://api.crossref.org/works"

CROSSREF_MESSAGE = {
    "DOI": "10.1000/synth.2025",
    "title": ["Crossref Paper Title"],
    "author": [{"given": "Ana", "family": "Svensson"}, {"given": "Bo", "family": "Chen"}],
    "issued": {"date-parts": [[2025, 3, 1]]},
    "container-title": ["Journal of Synthetic Studies"],
    "subject": ["crop"],
    "abstract": "A crossref abstract.",
}

CROSSREF_MESSAGE_NO_ABSTRACT = {
    "DOI": "10.1000/synth.2025",
    "title": ["Crossref Paper Title"],
    "author": [{"given": "Ana", "family": "Svensson"}],
    "issued": {"date-parts": [[2025, 3, 1]]},
    "container-title": ["Journal of Synthetic Studies"],
}

TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>
<titleStmt><title type="main">Grobid Paper Title</title></titleStmt>
<publicationStmt><publisher>Synth Press</publisher>
<date type="published" when="2018-06-15"/></publicationStmt>
<sourceDesc><biblStruct><analytic>
<title level="a">Grobid Paper Title</title>
<author><persName><forename>Liu</forename><surname>Tan</surname></persName></author>
</analytic><monogr><title level="j">Journal of Synthetic Studies</title>
<imprint><biblScope unit="volume">7</biblScope>
<idno type="DOI">10.9999/grobid.777</idno></imprint></monogr></biblStruct></sourceDesc>
</fileDesc><profileDesc><abstract><p>Grobid abstract body.</p></abstract></profileDesc></teiHeader></TEI>
"""

ABSTRACT_FILLED_MARKDOWN = (
    "# Heuristic Title\n\n2021\n\n## Abstract\n\nA useful abstract body.\n"
    "\n## Introduction\n\nIntro text follows here.\n"
)
NO_STRUCTURE_MARKDOWN = "This document has no markdown headings at all.\n\nJust text lines.\n"


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "false")


def _staged(tmp_path: Path, name: str = "paper_a") -> StagedPdf:
    source = Path(__file__).resolve().parents[2] / "fixtures" / f"{name}.pdf"
    return stage_pdf(source, tmp_path / "work")


def _crossref_ok(doi: str) -> None:
    respx.get(f"{CROSSREF_WORKS}/{doi}").mock(
        return_value=httpx.Response(200, json={"message": CROSSREF_MESSAGE})
    )


def test_unresolved_when_all_sources_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: None)
    result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "unresolved"
    assert result.title == "paper_a"
    assert result.title


def test_local_heuristic_extracts_title_year_and_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: None)
    markdown = "# Discovered Title\n\nJames Liao, Maya Patel, and Priya Nair\n\n2021\n\n## Introduction\n\nBody text."
    result = metadata_mod.stage_metadata(staged, markdown)
    assert result.source == "local"
    assert result.title == "Discovered Title"
    assert result.year == 2021
    assert result.authors == ["James Liao", "Maya Patel", "Priya Nair"]


def test_crossref_path_when_heuristic_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.1000/synth.2025")
    with respx.mock:
        _crossref_ok("10.1000/synth.2025")
        result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "crossref"
    assert result.title == "Crossref Paper Title"
    assert result.year == 2025
    assert result.authors == ["Ana Svensson", "Bo Chen"]
    assert result.journal == "Journal of Synthetic Studies"
    assert result.doi == "10.1000/synth.2025"


def test_crossref_failure_falls_to_grobid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.9999/grobid.777")
    monkeypatch.setenv("SCHOLAR_RAG_GROBID_URL", "http://grobid.test")
    with respx.mock:
        respx.get(f"{CROSSREF_WORKS}/10.9999/grobid.777").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        respx.post("http://grobid.test/api/processHeaderDocument").mock(
            return_value=httpx.Response(200, text=TEI_XML)
        )
        result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "grobid"
    assert result.title == "Grobid Paper Title"
    assert result.abstract == "Grobid abstract body."


def test_grobid_skipped_when_not_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.1000/synth.2025")
    with respx.mock:
        respx.get(f"{CROSSREF_WORKS}/10.1000/synth.2025").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "unresolved"


def test_stage_metadata_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)

    def boom(pdf: Path) -> str:
        raise RuntimeError("pdf2doi exploded")

    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", boom)
    result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "unresolved"
    assert result.title


def test_crossref_disabled_skips_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_CROSSREF_ENABLED", "false")
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.1000/synth.2025")
    with respx.mock:
        route = respx.get(f"{CROSSREF_WORKS}/10.1000/synth.2025").mock(
            return_value=httpx.Response(200, json={"message": CROSSREF_MESSAGE})
        )
        result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert route.called is False
    assert result.source == "unresolved"


def test_doc_id_is_deterministic() -> None:
    meta = MetadataResult(
        title="Some Title",
        year=2020,
        journal=None,
        doi=None,
        authors=["A"],
        abstract=None,
        source="local",
    )
    first = metadata_mod.build_doc_id(meta, "paper.pdf")
    second = metadata_mod.build_doc_id(meta, "paper.pdf")
    assert first == second
    assert first.startswith("doc-")
    different = metadata_mod.build_doc_id(meta, "other.pdf")
    assert different != first


def test_doc_id_prefers_doi() -> None:
    with_doi = MetadataResult(
        title="Some Title",
        year=2020,
        journal=None,
        doi="10.1000/abc.123",
        authors=[],
        abstract=None,
        source="crossref",
    )
    without_doi = MetadataResult(
        title="Some Title",
        year=2020,
        journal=None,
        doi=None,
        authors=[],
        abstract=None,
        source="local",
    )
    assert metadata_mod.build_doc_id(with_doi, "paper.pdf") == metadata_mod.build_doc_id(
        with_doi, "other.pdf"
    )
    assert metadata_mod.build_doc_id(with_doi, "paper.pdf") != metadata_mod.build_doc_id(
        without_doi, "paper.pdf"
    )


class FakeChatClient:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, *, json_schema: dict, max_retries: int = 2):
        self.calls.append(prompt)
        return self._response


def _fake_abstract_enricher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metadata_mod,
        "AbstractEnricher",
        lambda: SimpleNamespaceEnricher(),
    )


def test_keywords_generated_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "true")
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_BASE_URL", "http://chat.test/v1")
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: None)
    _fake_abstract_enricher(monkeypatch)
    fake = FakeChatClient({"keywords": ["yield forecasting", "remote sensing", "weather data", "crop model", "time series", "plant growth", "soil moisture", "harvest estimation"]})
    monkeypatch.setattr(metadata_mod, "get_chat_client", lambda: fake)
    result = metadata_mod.stage_metadata(staged, ABSTRACT_FILLED_MARKDOWN)
    assert result.source == "local"
    assert "yield forecasting" in result.keywords
    assert fake.calls


def test_keywords_disabled_yields_empty_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "false")
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: None)
    _fake_abstract_enricher(monkeypatch)
    monkeypatch.setattr(metadata_mod, "get_chat_client", lambda: FakeChatClient({}))
    result = metadata_mod.stage_metadata(staged, ABSTRACT_FILLED_MARKDOWN)
    assert result.keywords == ()


def test_keywords_failure_returns_empty_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "true")
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_BASE_URL", "http://chat.test/v1")
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: None)
    _fake_abstract_enricher(monkeypatch)

    class BrokenChat(FakeChatClient):
        def complete(self, prompt: str, *, json_schema: dict, max_retries: int = 2):
            raise RuntimeError("chat down")

    monkeypatch.setattr(metadata_mod, "get_chat_client", lambda: BrokenChat({}))
    result = metadata_mod.stage_metadata(staged, ABSTRACT_FILLED_MARKDOWN)
    assert result.title == "Heuristic Title"
    assert result.keywords == ()


@respx.mock
def test_abstract_enriched_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.1000/synth.2025")
    respx.get(f"{CROSSREF_WORKS}/10.1000/synth.2025").mock(
        return_value=httpx.Response(200, json={"message": CROSSREF_MESSAGE_NO_ABSTRACT})
    )
    monkeypatch.setattr(
        metadata_mod,
        "AbstractEnricher",
        lambda: SimpleNamespaceEnricher(),
    )
    result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.source == "crossref"
    assert result.abstract == "enriched abstract text"


@respx.mock
def test_abstract_kept_when_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = _staged(tmp_path)
    monkeypatch.setattr(metadata_mod.pdf2doi, "extract_doi", lambda pdf: "10.1000/synth.2025")
    _crossref_ok("10.1000/synth.2025")
    calls: list[dict] = []

    class RecordingEnricher(SimpleNamespaceEnricher):
        def enrich(self, metadata: dict) -> object:
            calls.append(metadata)
            return SimpleEnriched(abstract="should not apply", source="fake")

    monkeypatch.setattr(metadata_mod, "AbstractEnricher", lambda: RecordingEnricher())
    result = metadata_mod.stage_metadata(staged, NO_STRUCTURE_MARKDOWN)
    assert result.abstract == "A crossref abstract."
    assert calls == []


class SimpleNamespaceEnricher:
    def enrich(self, metadata: dict) -> object:
        return SimpleEnriched(abstract="enriched abstract text", source="fake")


class SimpleEnriched:
    def __init__(self, abstract: str, source: str) -> None:
        self.abstract = abstract
        self.source = source
