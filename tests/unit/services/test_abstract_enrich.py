from __future__ import annotations

import httpx
import respx

from scholar_rag.services.abstract_enrich import (
    AbstractEnricher,
    normalize_abstract_text,
    reconstruct_openalex_abstract,
)

_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_OPENALEX = "https://api.openalex.org/works"

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle><MedlineCitation>
<Article><Abstract>
<AbstractText Label="BACKGROUND">Background section text.</AbstractText>
<AbstractText>Plain methods section text.</AbstractText>
</Abstract></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_normalize_abstract_text() -> None:
    raw = "<jats:p>Hello &amp; <i>world</i>  more</jats:p>"
    assert normalize_abstract_text(raw) == "Hello & world more"


def test_reconstruct_openalex_abstract() -> None:
    index = {
        "the": [0, 5],
        "model": [1],
        "performs": [2],
        "well": [3],
    }
    assert reconstruct_openalex_abstract(index) == "the model performs well the"


def test_reconstruct_openalex_abstract_empty() -> None:
    assert reconstruct_openalex_abstract(None) == ""
    assert reconstruct_openalex_abstract({}) == ""


@respx.mock
def test_europe_pmc_source() -> None:
    respx.get(_EUROPE_PMC).mock(
        return_value=httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "title": "A Real Title",
                            "pubYear": "2020",
                            "abstractText": "<p>The abstract from europe pmc.</p>",
                        }
                    ]
                }
            },
        )
    )
    candidate = AbstractEnricher().enrich(
        {"doi": "", "title": "A Real Title", "year": 2020, "abstract": ""}
    )
    assert candidate is not None
    assert candidate.source == "europe_pmc"
    assert candidate.abstract == "The abstract from europe pmc."


@respx.mock
def test_pubmed_source_via_doi() -> None:
    respx.get(_EUROPE_PMC).mock(return_value=httpx.Response(200, json={"resultList": {"result": []}}))
    respx.get(_PUBMED_ESEARCH).mock(
        return_value=httpx.Response(
            200, json={"esearchresult": {"idlist": ["24680"]}}
        )
    )
    respx.get(_PUBMED_EFETCH).mock(return_value=httpx.Response(200, text=PUBMED_XML))
    candidate = AbstractEnricher().enrich(
        {"doi": "10.1000/pub.0001", "title": "", "year": None, "abstract": ""}
    )
    assert candidate is not None
    assert candidate.source == "pubmed"
    assert candidate.abstract == "Background section text.\nPlain methods section text."


@respx.mock
def test_openalex_source_with_inverted_index() -> None:
    respx.get(_EUROPE_PMC).mock(
        return_value=httpx.Response(200, json={"resultList": {"result": []}})
    )
    respx.get(_PUBMED_ESEARCH).mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
    )
    respx.get(_OPENALEX).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "display_name": "An OpenAlex Paper",
                        "publication_year": 2019,
                        "abstract_inverted_index": {
                            "modern": [0],
                            "retrieval": [1],
                            "works": [2],
                        },
                    }
                ]
            },
        )
    )
    candidate = AbstractEnricher().enrich(
        {"doi": "", "title": "An OpenAlex Paper", "year": 2019, "abstract": ""}
    )
    assert candidate is not None
    assert candidate.source == "openalex"
    assert candidate.abstract == "modern retrieval works"


@respx.mock
def test_title_mismatch_is_rejected() -> None:
    respx.get(_EUROPE_PMC).mock(
        return_value=httpx.Response(
            200,
            json={
                "resultList": {
                    "result": [
                        {
                            "title": "Unrelated Paper",
                            "pubYear": "2020",
                            "abstractText": "some abstract",
                        }
                    ]
                }
            },
        )
    )
    respx.get(_PUBMED_ESEARCH).mock(
        return_value=httpx.Response(200, json={"esearchresult": {"idlist": []}})
    )
    respx.get(_OPENALEX).mock(return_value=httpx.Response(200, json={"results": []}))
    candidate = AbstractEnricher().enrich(
        {"doi": "", "title": "A Completely Different Title", "year": 2020, "abstract": ""}
    )
    assert candidate is None


@respx.mock
def test_all_sources_failure_returns_none() -> None:
    for url in (_EUROPE_PMC, _PUBMED_ESEARCH, _OPENALEX):
        respx.get(url).mock(side_effect=httpx.ConnectError("connection refused"))
    candidate = AbstractEnricher().enrich(
        {"doi": "10.1000/x.1", "title": "Some Title", "year": 2020, "abstract": ""}
    )
    assert candidate is None
