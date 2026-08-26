from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_REQUEST_TIMEOUT = 60


class GrobidClient:
    def __init__(self, grobid_url: str) -> None:
        self.grobid_url = grobid_url.rstrip("/")
        self._header_api = f"{self.grobid_url}/api/processHeaderDocument"

    def extract(self, pdf_path: Path) -> dict[str, object] | None:
        try:
            with pdf_path.open("rb") as handle:
                resp = httpx.post(
                    self._header_api,
                    files={"input": (pdf_path.name, handle, "application/pdf")},
                    headers={"Accept": "application/xml"},
                    timeout=_REQUEST_TIMEOUT,
                )
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        return self.parse_tei_xml(resp.text)

    @staticmethod
    def parse_tei_xml(xml_text: str) -> dict[str, object]:
        result: dict[str, object] = {
            "title": "",
            "authors": [],
            "year": None,
            "abstract": "",
            "doi": "",
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "publisher": "",
            "keywords": [],
        }

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return result

        title_el = root.find(".//tei:titleStmt/tei:title[@type='main']", _TEI_NS)
        if title_el is None:
            title_el = root.find(".//tei:titleStmt/tei:title", _TEI_NS)
        if title_el is not None and title_el.text:
            result["title"] = title_el.text.strip()

        authors: list[str] = []
        for author in root.findall(".//tei:fileDesc//tei:author", _TEI_NS):
            persname = author.find("tei:persName", _TEI_NS)
            if persname is None:
                continue
            forename_parts: list[str] = []
            for fn in persname.findall("tei:forename", _TEI_NS):
                if fn.text:
                    forename_parts.append(fn.text.strip())
            surname_el = persname.find("tei:surname", _TEI_NS)
            surname_text = ""
            if surname_el is not None and surname_el.text:
                surname_text = surname_el.text.strip()
            if forename_parts and surname_text:
                authors.append(f"{' '.join(forename_parts)} {surname_text}")
            elif surname_text:
                authors.append(surname_text)
        result["authors"] = authors

        date_el = root.find(".//tei:publicationStmt/tei:date[@type='published']", _TEI_NS)
        if date_el is None:
            date_el = root.find(".//tei:sourceDesc//tei:date[@type='published']", _TEI_NS)
        if date_el is None:
            date_el = root.find(".//tei:sourceDesc//tei:date", _TEI_NS)
        if date_el is not None:
            when = date_el.get("when", "")
            if when:
                year_match = re.match(r"(\d{4})", when)
                if year_match:
                    result["year"] = int(year_match.group(1))

        abstract_el = root.find(".//tei:profileDesc/tei:abstract", _TEI_NS)
        if abstract_el is not None:
            abstract_parts: list[str] = []
            for p in abstract_el.findall(".//tei:p", _TEI_NS):
                text = "".join(p.itertext()).strip()
                if text:
                    abstract_parts.append(text)
            if abstract_parts:
                result["abstract"] = " ".join(abstract_parts)
            else:
                text = "".join(abstract_el.itertext()).strip()
                if text:
                    result["abstract"] = text

        for idno in root.findall(".//tei:sourceDesc//tei:idno[@type='DOI']", _TEI_NS):
            if idno.text:
                result["doi"] = idno.text.strip().lower()
                break

        journal_el = root.find(
            ".//tei:sourceDesc//tei:monogr/tei:title[@level='j']",
            _TEI_NS,
        )
        if journal_el is not None and journal_el.text:
            result["journal"] = journal_el.text.strip()

        for biblscope in root.findall(".//tei:sourceDesc//tei:biblScope", _TEI_NS):
            unit = biblscope.get("unit", "")
            text = biblscope.text.strip() if biblscope.text else ""
            from_val = biblscope.get("from", "")
            to_val = biblscope.get("to", "")

            if unit == "volume" and (text or from_val):
                result["volume"] = text or from_val
            elif unit == "issue" and (text or from_val):
                result["issue"] = text or from_val
            elif unit == "page":
                if from_val and to_val:
                    result["pages"] = f"{from_val}-{to_val}"
                elif text:
                    result["pages"] = text

        publisher_el = root.find(".//tei:publicationStmt/tei:publisher", _TEI_NS)
        if publisher_el is not None and publisher_el.text:
            result["publisher"] = publisher_el.text.strip()

        keywords: list[str] = []
        for kw in root.findall(".//tei:profileDesc//tei:keywords//tei:term", _TEI_NS):
            if kw.text and kw.text.strip():
                keywords.append(kw.text.strip())
        result["keywords"] = keywords

        return result
