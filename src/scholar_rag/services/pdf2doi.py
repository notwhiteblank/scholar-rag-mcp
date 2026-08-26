from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType

from scholar_rag.services.crossref import clean_doi_deep, strip_component_suffix

_DOI_TEXT_PATTERN = re.compile(
    r"(?:doi[:\s]*|DOI[:\s]*|https?://(?:dx\.)?doi\.org/)?"
    r"(10\.\d{4,9}/[^\s]{2,})",
    re.IGNORECASE,
)
_TRAILING_JUNK = re.compile(r'[\s.,;:!?\'")\]}>]+$')
_TAG_PATTERN = re.compile(r"<[^>]+>")


def extract_doi(pdf_path: Path) -> str | None:
    doi, _method = extract_doi_with_info(pdf_path)
    return doi


def extract_doi_with_info(pdf_path: Path) -> tuple[str | None, str]:
    doi, method = _try_pdf2doi(pdf_path)
    if doi is None:
        doi, method = _try_regex(pdf_path)
    if doi is None:
        return None, "none"

    cleaned = clean_doi_deep(doi)
    if cleaned is None:
        return None, "none"

    article_doi = strip_component_suffix(cleaned)
    if article_doi is not None:
        return article_doi, f"{method} (component_fix)"
    return cleaned, method


def _try_pdf2doi(pdf_path: Path) -> tuple[str | None, str]:
    try:
        pdf2doi = _import_pdf2doi()
    except ImportError:
        return None, "none"

    try:
        pdf2doi.config.set("save_identifier_metadata", False)
        pdf2doi.config.set("verbose", False)
        pdf2doi.config.set("websearch", False)
        pdf2doi.config.set("webvalidation", False)
        result = pdf2doi.pdf2doi(str(pdf_path))
    except Exception:
        return None, "none"

    if isinstance(result, list):
        result = result[0] if result else None

    if isinstance(result, dict) and result.get("identifier"):
        method = result.get("method", "unknown")
        return str(result["identifier"]), f"pdf2doi ({method})"

    return None, "none"


def _import_pdf2doi() -> ModuleType:
    return importlib.import_module("pdf2doi")


def _try_regex(pdf_path: Path) -> tuple[str | None, str]:
    text = _extract_text_first_pages(pdf_path)
    if not text.strip():
        return None, "none"

    candidates = _extract_dois_from_text(text)
    best = _select_best_doi(candidates)

    if best:
        return best, "regex_first_page"
    return None, "none"


def _extract_text_first_pages(pdf_path: Path, max_pages: int = 2) -> str:
    try:
        fitz = _import_fitz()
    except ImportError:
        return ""

    try:
        doc = fitz.open(str(pdf_path))
        text_parts: list[str] = []
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception:
        return ""


def _import_fitz() -> ModuleType:
    return importlib.import_module("fitz")


def _extract_dois_from_text(text: str) -> list[str]:
    text_joined = re.sub(r"(10\.\d{4,9}/\S*)\s*\n\s*(\S+)", r"\1\2", text)

    candidates: list[str] = []
    for match in _DOI_TEXT_PATTERN.finditer(text_joined):
        cleaned = _clean_doi_regex(match.group(1))
        if _validate_doi_format(cleaned):
            candidates.append(cleaned)

    for match in _DOI_TEXT_PATTERN.finditer(text):
        cleaned = _clean_doi_regex(match.group(1))
        if _validate_doi_format(cleaned) and cleaned not in candidates:
            candidates.append(cleaned)

    return candidates


def _clean_doi_regex(raw_doi: str) -> str:
    doi = raw_doi.strip()
    doi = _TAG_PATTERN.sub("", doi)
    doi = re.sub(r"\s+", "", doi)
    doi = doi.replace("%20", "")

    for _ in range(5):
        old = doi
        doi = _TRAILING_JUNK.sub("", doi)
        if doi == old:
            break

    open_count = doi.count("(")
    close_count = doi.count(")")
    if open_count > close_count:
        last_open = doi.rfind("(")
        doi = doi[:last_open].rstrip(".")
    elif close_count > open_count:
        while doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = doi[:-1]

    doi = _TRAILING_JUNK.sub("", doi)

    return doi


def _validate_doi_format(doi: str) -> bool:
    if not doi or len(doi) < 8:
        return False
    if not doi.startswith("10."):
        return False
    if "/" not in doi[3:]:
        return False
    suffix = doi.split("/", 1)[1]
    return not len(suffix) < 2


def _select_best_doi(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return candidates[0]
