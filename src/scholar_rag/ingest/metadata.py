from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from typing import Any

from scholar_rag.core.config import Settings
from scholar_rag.core.types import MetadataResult
from scholar_rag.ingest.staging import StagedPdf
from scholar_rag.models import get_chat_client
from scholar_rag.services import crossref, pdf2doi
from scholar_rag.services.abstract_enrich import AbstractEnricher
from scholar_rag.services.grobid import GrobidClient

_MAX_YEAR_SNIFF_CHARS = 1000
_MAX_ABSTRACT_CHARS = 6000
_KEYWORDS_MIN = 2
_MAX_KEYWORDS = 10
_MAX_KEYWORD_CHARS = 80
_SPLIT_RE = re.compile(r"[\n;,|]+")
_LEADING_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s*")
_GENERIC_KEYWORDS = {
    "abstract",
    "article",
    "keyword",
    "keywords",
    "paper",
    "research",
    "review",
    "study",
}
_TITLE_HEADING_RE = re.compile(r"^#{1,2}\s+(.+)")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_HEADING_START_RE = re.compile(r"^#{1,6}\s+")
_AUTHOR_NAME_RE = re.compile(
    r"[A-Z][A-Za-z'`-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'`-]+)+(?:\d+|\*)?"
)


def stage_metadata(staged: StagedPdf, markdown: str) -> MetadataResult:
    settings = Settings.load()
    result = _extract_staged(staged, markdown, settings)
    return _enrich(result, settings)


def _extract_staged(
    staged: StagedPdf,
    markdown: str,
    settings: Settings,
) -> MetadataResult:
    try:
        heuristic = _try_local_heuristic(staged, markdown)
    except Exception:
        heuristic = None
    if heuristic is not None:
        return heuristic

    try:
        crossref_result = _try_crossref(staged, settings)
    except Exception:
        crossref_result = None
    if crossref_result is not None:
        return crossref_result

    try:
        grobid_result = _try_grobid(staged, settings)
    except Exception:
        grobid_result = None
    if grobid_result is not None:
        return grobid_result

    return MetadataResult(
        title=staged.name,
        year=None,
        journal=None,
        doi=None,
        authors=[],
        abstract=None,
        source="unresolved",
    )


def _try_local_heuristic(staged: StagedPdf, markdown: str) -> MetadataResult | None:
    del staged
    title = _extract_markdown_title(markdown)
    if not title:
        return None
    year = _extract_markdown_year(markdown)
    authors = _extract_markdown_authors(markdown)
    return MetadataResult(
        title=title,
        year=year,
        journal=None,
        doi=None,
        authors=authors,
        abstract=None,
        source="local",
    )


def _extract_markdown_title(markdown: str) -> str:
    for line in markdown.splitlines():
        match = _TITLE_HEADING_RE.match(line)
        if match:
            return " ".join(match.group(1).strip().split())
    return ""


def _extract_markdown_year(markdown: str) -> int | None:
    match = _YEAR_RE.search(markdown[:_MAX_YEAR_SNIFF_CHARS])
    if match:
        return int(match.group(0))
    return None


def _extract_markdown_authors(markdown: str) -> list[str]:
    blocks = [block.strip() for block in markdown.split("\n\n") if block.strip()]
    if not blocks:
        return []
    preamble = blocks[1:] if blocks[0].startswith("#") else blocks
    for block in preamble:
        if _HEADING_START_RE.match(block):
            break
        if _is_author_line(block):
            return _split_author_line(block)
    return []


def _is_author_line(text: str) -> bool:
    if not 10 <= len(text) <= 300:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    if len(_AUTHOR_NAME_RE.findall(text)) < 2:
        return False
    return "," in text or " and " in text.lower()


def _split_author_line(line: str) -> list[str]:
    parts = [part.strip().rstrip(".") for part in re.split(r",|(?:\s+and\s+)", line)]
    return [part for part in parts if part]


def _try_crossref(staged: StagedPdf, settings: Settings) -> MetadataResult | None:
    if not settings.crossref_enabled:
        return None
    doi = pdf2doi.extract_doi(staged.path)
    if not doi:
        return None
    if crossref.is_pdb_doi(doi):
        return None
    message = crossref.fetch_crossref_metadata(doi)
    if message is None:
        return None
    payload = crossref.parse_crossref_message(message, doi_method="crossref")
    return _metadata_from_payload(payload, source="crossref")


def _try_grobid(staged: StagedPdf, settings: Settings) -> MetadataResult | None:
    if not settings.grobid_url:
        return None
    client = GrobidClient(settings.grobid_url)
    payload = client.extract(staged.path)
    if payload is None:
        return None
    if not (payload.get("title") or payload.get("doi") or payload.get("authors")):
        return None
    return _metadata_from_payload(payload, source="grobid")


def _metadata_from_payload(
    payload: dict[str, object],
    *,
    source: str,
) -> MetadataResult:
    raw_authors = payload.get("authors")
    authors = (
        [str(item) for item in raw_authors if item] if isinstance(raw_authors, list) else []
    )
    year = payload.get("year")
    year_value = int(year) if isinstance(year, int) else None
    abstract = _clean_text(payload.get("abstract")) or None
    journal = _clean_text(payload.get("journal")) or None
    doi = _clean_text(payload.get("doi")) or None
    title = _clean_text(payload.get("title"))
    raw_keywords = payload.get("keywords")
    keywords = (
        tuple(str(item) for item in raw_keywords if item)
        if isinstance(raw_keywords, list)
        else ()
    )
    return MetadataResult(
        title=title,
        year=year_value,
        journal=journal,
        doi=doi,
        authors=authors,
        abstract=abstract,
        source=source,
        keywords=keywords,
    )


def _enrich(result: MetadataResult, settings: Settings) -> MetadataResult:
    if result.source != "unresolved" and result.abstract is None:
        result = _try_enrich_abstract(result)
    keywords = _generate_keywords(result, settings)
    if keywords and result.keywords != keywords:
        result = replace(result, keywords=keywords)
    return result


def _try_enrich_abstract(result: MetadataResult) -> MetadataResult:
    try:
        candidate = AbstractEnricher().enrich(
            {
                "doi": result.doi,
                "title": result.title,
                "year": result.year,
                "abstract": result.abstract,
            }
        )
    except Exception:
        return result
    if candidate is None or not candidate.abstract:
        return result
    return replace(result, abstract=candidate.abstract)


def _chat_configured(settings: Settings) -> bool:
    return bool(settings.chat_base_url)


def _generate_keywords(result: MetadataResult, settings: Settings) -> tuple[str, ...]:
    if not settings.keywords_enabled:
        return ()
    if not result.abstract or not result.title:
        return ()
    if not _chat_configured(settings):
        return ()
    try:
        client = get_chat_client()
    except Exception:
        return ()
    prompt = _build_keywords_prompt(result.title, result.abstract)
    try:
        payload = client.complete(prompt, json_schema=keywords_schema(), max_retries=2)
    except Exception:
        return ()
    if not isinstance(payload, dict):
        return ()
    keywords = normalize_keyword_list(payload.get("keywords"))
    if len(keywords) < _KEYWORDS_MIN:
        return ()
    return tuple(keywords)


def normalize_keyword_list(value: Any) -> list[str]:
    raw_items: list[str] = []
    if isinstance(value, list):
        for item in value:
            raw_items.extend(normalize_keyword_list(item))
    else:
        text = _clean_text(value)
        if not text:
            return []
        if any(sep in text for sep in ("\n", ";", ",")):
            raw_items.extend(part for part in _SPLIT_RE.split(text) if part.strip())
        else:
            raw_items.append(text)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _clean_text(item)
        text = _LEADING_BULLET_RE.sub("", text).strip(" .,:;/-_")
        if not text:
            continue
        if len(text) > _MAX_KEYWORD_CHARS:
            continue
        if text.lower() in _GENERIC_KEYWORDS:
            continue
        dedup_key = re.sub(r"[^a-z0-9]+", "", text.lower())
        if not dedup_key or dedup_key in seen:
            continue
        seen.add(dedup_key)
        normalized.append(text)
        if len(normalized) >= _MAX_KEYWORDS:
            break
    return normalized


def keywords_schema() -> dict[str, Any]:
    return {
        "title": "KeywordResponse",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["keywords"],
    }


def _build_keywords_prompt(title: str, abstract: str) -> str:
    clipped = abstract[:_MAX_ABSTRACT_CHARS]
    return (
        "Generate 8-10 English metadata keywords for academic document indexing and retrieval.\n\n"
        'Return a json object with a single key: "keywords".\n\n'
        "Requirements:\n"
        "- Keywords are for document-level metadata filtering and lexical recall, not for summarization.\n"
        "- Return short keywords or noun phrases only, not sentences.\n"
        "- Most keywords should be 1-4 words. Use longer phrases only for standard technical names.\n"
        "- Prefer exact present terms from the title and abstract when they are specific and retrieval-useful.\n"
        "- Preserve canonical abbreviations and hyphenation from the paper when available.\n"
        "- Avoid generic fillers like study, article, review, research unless they are part of a standard technical phrase.\n"
        "- Remove near-duplicates and avoid paraphrase variants that do not improve retrieval.\n\n"
        f"Title:\n{title}\n\n"
        f"Abstract:\n{clipped}\n"
    )


def build_doc_id(meta: MetadataResult, source_file: str) -> str:
    doi = _lookup_text(meta.doi).lower()
    title_norm = _lookup_text(meta.title)
    source_norm = _lookup_text(source_file)
    if doi:
        basis = f"doi:{doi}"
    elif title_norm:
        basis = f"title:{title_norm}|source:{source_norm}"
    else:
        basis = f"source:{source_norm}"
    digest = hashlib.blake2b(basis.encode("utf-8"), digest_size=10).hexdigest()
    return f"doc-{digest}"


def _lookup_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().casefold()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()
