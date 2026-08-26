from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from scholar_rag.services.crossref import title_similarity

REQUEST_TIMEOUT = 10
_MIN_TITLE_SIMILARITY = 0.9

_EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_OPENALEX_WORKS = "https://api.openalex.org/works"

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

_HEADERS = {
    "User-Agent": "ScholarRagMCP/0.1 (academic metadata enrichment)"
}


def normalize_abstract_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def reconstruct_openalex_abstract(
    inverted_index: Any,
) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""

    max_pos = -1
    for positions in inverted_index.values():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                max_pos = max(max_pos, pos)
    if max_pos < 0:
        return ""

    words = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos < len(words):
                words[pos] = token

    text = " ".join(word for word in words if word)
    return normalize_abstract_text(text)


@dataclass(frozen=True)
class AbstractCandidate:
    abstract: str
    source: str


class AbstractEnricher:
    def enrich(self, metadata: dict[str, Any]) -> AbstractCandidate | None:
        doi = self._normalize_doi(metadata.get("doi"))
        title = normalize_abstract_text(metadata.get("title"))
        year = self._normalize_year(metadata.get("year"))

        for candidate in self._iter_candidates(doi=doi, title=title, year=year):
            if candidate is not None and candidate.abstract:
                return candidate
        return None

    def _iter_candidates(
        self,
        *,
        doi: str,
        title: str,
        year: int | None,
    ) -> Generator[AbstractCandidate | None, None, None]:
        yield self._fetch_europe_pmc(doi=doi, title=title, year=year)
        yield self._fetch_pubmed(doi=doi)
        yield self._fetch_openalex(doi=doi, title=title, year=year)

    def _fetch_europe_pmc(
        self,
        *,
        doi: str,
        title: str,
        year: int | None,
    ) -> AbstractCandidate | None:
        if doi:
            query = f'DOI:"{doi}"'
        elif title:
            query = f'TITLE:"{title}"'
        else:
            return None

        try:
            resp = httpx.get(
                _EUROPE_PMC_SEARCH,
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": 5,
                    "resultType": "core",
                },
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        result_list = data.get("resultList", {}) if isinstance(data, dict) else {}
        results = result_list.get("result", []) if isinstance(result_list, dict) else []
        if not isinstance(results, list):
            return None
        for item in results:
            candidate = self._candidate_from_europe_pmc(item, title=title, year=year)
            if candidate is not None:
                return candidate
        return None

    def _candidate_from_europe_pmc(
        self,
        item: object,
        *,
        title: str,
        year: int | None,
    ) -> AbstractCandidate | None:
        if not isinstance(item, dict):
            return None
        abstract = normalize_abstract_text(item.get("abstractText"))
        if not abstract:
            return None

        item_title = normalize_abstract_text(item.get("title"))
        if title and item_title:
            sim = title_similarity(title, item_title)
            if sim < _MIN_TITLE_SIMILARITY:
                return None

        if year is not None:
            item_year = self._normalize_year(item.get("pubYear"))
            if item_year is not None and abs(item_year - year) > 1:
                return None

        return AbstractCandidate(abstract=abstract, source="europe_pmc")

    def _fetch_pubmed(self, *, doi: str) -> AbstractCandidate | None:
        if not doi:
            return None

        pmid = self._lookup_pubmed_id(doi)
        if not pmid:
            return None

        try:
            resp = httpx.get(
                _PUBMED_EFETCH,
                params={
                    "db": "pubmed",
                    "id": pmid,
                    "retmode": "xml",
                },
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except (httpx.HTTPError, ValueError):
            return None

        abstract = self._parse_pubmed_xml_abstract(resp.text)
        if not abstract:
            return None
        return AbstractCandidate(abstract=abstract, source="pubmed")

    def _lookup_pubmed_id(self, doi: str) -> str:
        try:
            resp = httpx.get(
                _PUBMED_ESEARCH,
                params={
                    "db": "pubmed",
                    "term": f"{doi}[AID]",
                    "retmode": "json",
                    "retmax": 1,
                },
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return ""

        if not isinstance(data, dict):
            return ""
        esearch_result = data.get("esearchresult")
        if not isinstance(esearch_result, dict):
            return ""
        ids = esearch_result.get("idlist")
        if not isinstance(ids, list) or not ids:
            return ""
        return str(ids[0]).strip()

    def _fetch_openalex(
        self,
        *,
        doi: str,
        title: str,
        year: int | None,
    ) -> AbstractCandidate | None:
        if doi:
            params: dict[str, Any] = {
                "filter": f"doi:https://doi.org/{doi}",
                "per-page": 1,
            }
        elif title:
            params = {
                "search": title,
                "per-page": 5,
            }
        else:
            return None

        try:
            resp = httpx.get(
                _OPENALEX_WORKS,
                params=params,
                headers=_HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        results = data.get("results", []) if isinstance(data, dict) else []
        if not isinstance(results, list):
            return None
        for item in results:
            candidate = self._candidate_from_openalex(item, title=title, year=year)
            if candidate is not None:
                return candidate
        return None

    def _candidate_from_openalex(
        self,
        item: object,
        *,
        title: str,
        year: int | None,
    ) -> AbstractCandidate | None:
        if not isinstance(item, dict):
            return None

        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        if not abstract:
            return None

        item_title = normalize_abstract_text(item.get("display_name"))
        if title and item_title:
            sim = title_similarity(title, item_title)
            if sim < _MIN_TITLE_SIMILARITY:
                return None

        if year is not None:
            item_year = self._normalize_year(item.get("publication_year"))
            if item_year is not None and abs(item_year - year) > 1:
                return None

        return AbstractCandidate(abstract=abstract, source="openalex")

    @staticmethod
    def _parse_pubmed_xml_abstract(xml_text: str) -> str:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""

        parts: list[str] = []
        for node in root.findall(".//AbstractText"):
            text = normalize_abstract_text("".join(node.itertext()))
            if not text:
                continue
            label = normalize_abstract_text(node.attrib.get("Label"))
            if label and not text.lower().startswith(label.lower()):
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _normalize_doi(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        if text.startswith("https://doi.org/"):
            text = text.split("https://doi.org/", 1)[1]
        if text.startswith("http://doi.org/"):
            text = text.split("http://doi.org/", 1)[1]
        return quote(text, safe="/().-_:;")

    @staticmethod
    def _normalize_year(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return None
