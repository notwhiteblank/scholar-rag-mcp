from __future__ import annotations

import httpx
import respx

from scholar_rag.services.crossref import (
    clean_doi_deep,
    extract_title_from_filename,
    extract_year_from_filename,
    fetch_crossref_metadata,
    normalize_title,
    parse_crossref_message,
    strip_component_suffix,
    title_similarity,
)


def test_clean_doi_deep_removes_trailing_junk() -> None:
    assert clean_doi_deep("10.1000/abc.123",) == "10.1000/abc.123"
    assert clean_doi_deep("10.1000/abc.123.") == "10.1000/abc.123"
    assert clean_doi_deep("10.1000/abc.123 OPEN") == "10.1000/abc.123"


def test_clean_doi_deep_rejects_template_or_truncated() -> None:
    assert clean_doi_deep("10.1074/jbc") is None
    assert clean_doi_deep("10.xxxx/abcd") is None
    assert clean_doi_deep("not-a-doi") is None
    assert clean_doi_deep("10.1/x") is None


def test_clean_doi_deep_removes_url_path_suffix() -> None:
    assert clean_doi_deep("10.1000/abc.123/1752421/blood.pdf") == "10.1000/abc.123"


def test_strip_component_suffix_for_elife_and_plos() -> None:
    assert strip_component_suffix("10.7554/eLife.06807.001") == "10.7554/eLife.06807"
    assert strip_component_suffix("10.1371/journal.pcbi.1004013.t001") == "10.1371/journal.pcbi.1004013"
    assert strip_component_suffix("10.1000/plain.123") is None


def test_normalize_title_and_similarity() -> None:
    assert normalize_title("  Crop &amp; Yield Forecasting!  ") == "crop yield forecasting"
    assert title_similarity("Crop Yield Forecasting", "crop yield forecasting") == 1.0
    assert title_similarity("In Vitro Assay", "Complete Difference") < 0.3


def test_title_fieldname_extraction() -> None:
    assert extract_title_from_filename("2021-Some Study.pdf") == "Some Study"
    assert extract_year_from_filename("2021-Some Study.pdf") == 2021
    assert extract_year_from_filename("Some Study.pdf") is None


@respx.mock
def test_fetch_crossref_metadata_returns_message() -> None:
    route = respx.get("https://api.crossref.org/works/10.1000/abc.123").mock(
        return_value=httpx.Response(200, json={"message": {"DOI": "10.1000/abc.123", "title": ["T"]}})
    )
    data = fetch_crossref_metadata("10.1000/abc.123")
    assert data == {"DOI": "10.1000/abc.123", "title": ["T"]}
    assert route.call_count == 1


@respx.mock
def test_fetch_crossref_metadata_not_found_returns_none() -> None:
    respx.get("https://api.crossref.org/works/10.1000/missing").mock(
        return_value=httpx.Response(404, text="not found")
    )
    assert fetch_crossref_metadata("10.1000/missing") is None


@respx.mock
def test_fetch_crossref_metadata_network_error_returns_none() -> None:
    respx.get("https://api.crossref.org/works/10.1000/down").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    assert fetch_crossref_metadata("10.1000/down") is None


@respx.mock
def test_fetch_crossref_metadata_timeout_returns_none() -> None:
    respx.get("https://api.crossref.org/works/10.1000/slow").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    assert fetch_crossref_metadata("10.1000/slow") is None


def test_parse_crossref_message() -> None:
    message = {
        "DOI": "10.1000/abc.123",
        "title": ["Paper Title"],
        "author": [
            {"given": "Jane", "family": "Doe"},
            {"given": "Kai", "family": "Lee"},
        ],
        "issued": {"date-parts": [[2020, 5, 1]]},
        "container-title": ["Journal of Example"],
        "subject": ["crop"],
        "abstract": "An abstract",
    }
    metadata = parse_crossref_message(message, doi_method="regex_first_page")
    assert metadata["title"] == "Paper Title"
    assert metadata["year"] == 2020
    assert metadata["authors"] == ["Jane Doe", "Kai Lee"]
    assert metadata["journal"] == "Journal of Example"
    assert metadata["keywords"] == ["crop"]
    assert metadata["abstract"] == "An abstract"


def test_parse_crossref_message_defensive_shapes() -> None:
    metadata = parse_crossref_message({}, doi_method="none")
    assert metadata["title"] == ""
    assert metadata["authors"] == []
    assert metadata["year"] is None
    assert metadata["journal"] == ""
