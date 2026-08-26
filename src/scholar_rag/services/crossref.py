from __future__ import annotations

import re
from difflib import SequenceMatcher

import httpx

CROSSREF_API = "https://api.crossref.org/works/"
CROSSREF_HEADERS = {
    "User-Agent": "ScholarRagMCP/0.1 (academic metadata enrichment)"
}
REQUEST_TIMEOUT = 10
TITLE_MATCH_THRESHOLD = 0.6

JOURNAL_DOI_PREFIXES = {
    "10.1074/jbc",
    "10.1074/mcp",
    "10.1083/jcb",
    "10.1073/pnas",
    "10.1084/jem",
    "10.3390/ijms",
    "10.3892/mmr",
}

TRAILING_WORD_JUNK = re.compile(
    r"(?:OPEN|JSTOR|REVIEW|RESEARCH|Solution|Author|Address|"
    r"All|Europe|1CAS|CAS|ACCESS|ARTICLE|LETTER|PAPER|REPORT|"
    r"[A-Z]{2,})"
    r"$"
)
TRAILING_URL_PATH = re.compile(r"/\d+/[^/]+\.pdf$")
TRAILING_EXTRA_PATH = re.compile(r"/\d{5,}$")
TRAILING_PAREN_JUNK = re.compile(r"\)[.)*\w]*$")
UNICODE_JUNK = re.compile(r"[\ufffd\ufffe\uffff]+$")

_ELIFE_DOI_RE = re.compile(r"(10\.7554/elife\.\d+)", re.IGNORECASE)
_PLOS_COMPONENT_RE = re.compile(r"(10\.1371/journal\.\w+\.\d+)\.[tgse]\d+$")
_PDB_DOI_RE = re.compile(r"10\.2210/pdb", re.IGNORECASE)


def extract_title_from_filename(filename: str) -> str:
    name = filename.rsplit(".pdf", 1)[0]
    match = re.match(r"^\d{4}-(.+)$", name)
    if match:
        return match.group(1).strip()
    return name.strip()


def extract_year_from_filename(filename: str) -> int | None:
    match = re.match(r"^(\d{4})-", filename)
    if match:
        return int(match.group(1))
    return None


def normalize_title(title: str) -> str:
    title = title.lower()
    title = title.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def title_similarity(title1: str, title2: str) -> float:
    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if not norm1 or not norm2:
        return 0.0

    ratio = SequenceMatcher(None, norm1, norm2).ratio()

    if norm1 in norm2 or norm2 in norm1:
        ratio = max(ratio, 0.8)

    return ratio


def clean_doi_deep(doi: str) -> str | None:
    if not doi:
        return None

    doi = UNICODE_JUNK.sub("", doi)
    doi = TRAILING_URL_PATH.sub("", doi)
    doi = TRAILING_EXTRA_PATH.sub("", doi)

    paren_match = TRAILING_PAREN_JUNK.search(doi)
    if paren_match:
        prefix = doi[: paren_match.start()]
        if prefix.count("(") <= prefix.count(")"):
            doi = prefix

    for _ in range(3):
        old = doi
        doi = TRAILING_WORD_JUNK.sub("", doi)
        doi = re.sub(r"\.[A-Z][a-z]+$", "", doi)
        if doi == old:
            break

    doi = re.sub(r"[.,;:!?\s]+$", "", doi)

    if doi in JOURNAL_DOI_PREFIXES:
        return None

    if "cXsm00000x" in doi or "xxxx" in doi.lower():
        return None

    if not doi.startswith("10.") or "/" not in doi[3:]:
        return None
    suffix = doi.split("/", 1)[1]
    if len(suffix) < 2:
        return None

    return doi


def strip_component_suffix(doi: str) -> str | None:
    m = _ELIFE_DOI_RE.match(doi)
    if m:
        article_doi = m.group(1)
        if article_doi != doi:
            return article_doi

    m = _PLOS_COMPONENT_RE.match(doi)
    if m:
        return m.group(1)

    return None


def is_pdb_doi(doi: str) -> bool:
    return bool(_PDB_DOI_RE.match(doi))


def fetch_crossref_metadata(doi: str) -> dict[str, object] | None:
    url = f"{CROSSREF_API}{doi}"
    try:
        resp = httpx.get(url, headers=CROSSREF_HEADERS, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    message = data.get("message") if isinstance(data, dict) else None
    return message if isinstance(message, dict) else None


def parse_crossref_message(message: dict[str, object], doi_method: str) -> dict[str, object]:
    authors: list[str] = []
    raw_authors = message.get("author")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if isinstance(author, str):
                if author.strip():
                    authors.append(author.strip())
                continue
            if not isinstance(author, dict):
                continue
            given = author.get("given")
            family = author.get("family")
            given_text = str(given) if isinstance(given, str) else ""
            family_text = str(family) if isinstance(family, str) else ""
            if given_text and family_text:
                authors.append(f"{given_text} {family_text}")
            elif family_text:
                authors.append(family_text)
            elif given_text:
                authors.append(given_text)

    year: int | None = None
    for date_field in [
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    ]:
        date_info = message.get(date_field)
        if not isinstance(date_info, dict):
            continue
        date_parts = date_info.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            first = date_parts[0][0]
            if isinstance(first, int):
                year = first
                break

    title_raw = message.get("title")
    if isinstance(title_raw, list):
        title = str(title_raw[0]) if title_raw else ""
    elif isinstance(title_raw, str):
        title = title_raw
    else:
        title = ""

    keywords_raw = message.get("subject")
    keywords = keywords_raw if isinstance(keywords_raw, list) else []
    keywords = [str(item) for item in keywords if isinstance(item, str)]

    container = message.get("container-title")
    if isinstance(container, list) and container:
        journal = str(container[0])
    elif isinstance(container, str):
        journal = container
    else:
        journal = ""

    doi = message.get("DOI")
    abstract = message.get("abstract")

    return {
        "doi": str(doi) if isinstance(doi, str) else "",
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": str(abstract) if isinstance(abstract, str) else "",
        "journal": journal,
        "volume": str(message.get("volume") or ""),
        "issue": str(message.get("issue") or ""),
        "pages": str(message.get("page") or ""),
        "publisher": str(message.get("publisher") or ""),
        "keywords": keywords,
        "source": doi_method,
    }
