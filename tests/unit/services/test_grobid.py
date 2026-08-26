from __future__ import annotations

from pathlib import Path

import httpx
import respx

from scholar_rag.services.grobid import GrobidClient

TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">Grobid Extracted Title</title>
      </titleStmt>
      <publicationStmt>
        <publisher>Synth Press</publisher>
        <date type="published" when="2019-05-01"/>
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <title level="a">Grobid Extracted Title</title>
            <author><persName><forename>Jane</forename><forename>R.</forename><surname>Doe</surname></persName></author>
            <author><persName><forename>Qing</forename><surname>Wei</surname></persName></author>
          </analytic>
          <monogr>
            <title level="j">Journal of Synthetic Studies</title>
            <imprint>
              <biblScope unit="volume">12</biblScope>
              <biblScope unit="issue">3</biblScope>
              <biblScope unit="page" from="100" to="120"/>
              <idno type="DOI">10.9999/grobid.12345</idno>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>Grobid abstract paragraph one.</p>
        <p>Grobid abstract paragraph two.</p>
      </abstract>
      <textClass>
        <keywords><term>crop</term><term>yield</term></keywords>
      </textClass>
    </profileDesc>
  </teiHeader>
</TEI>
"""


def test_parse_tei_xml_extracts_metadata() -> None:
    meta = GrobidClient("http://grobid.test").parse_tei_xml(TEI_XML)
    assert meta["title"] == "Grobid Extracted Title"
    assert meta["authors"] == ["Jane R. Doe", "Qing Wei"]
    assert meta["year"] == 2019
    assert meta["abstract"] == "Grobid abstract paragraph one. Grobid abstract paragraph two."
    assert meta["doi"] == "10.9999/grobid.12345"
    assert meta["journal"] == "Journal of Synthetic Studies"
    assert meta["volume"] == "12"
    assert meta["issue"] == "3"
    assert meta["pages"] == "100-120"
    assert meta["publisher"] == "Synth Press"
    assert meta["keywords"] == ["crop", "yield"]


def test_parse_tei_xml_invalid_returns_defaults() -> None:
    meta = GrobidClient("http://grobid.test").parse_tei_xml("<not-tei>")
    assert meta["title"] == ""
    assert meta["authors"] == []
    assert meta["year"] is None


@respx.mock
def test_extract_posts_bytes_and_parses(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    respx.post("http://grobid.test/api/processHeaderDocument").mock(
        return_value=httpx.Response(200, text=TEI_XML, headers={"Content-Type": "application/xml"})
    )
    meta = GrobidClient("http://grobid.test").extract(pdf)
    assert meta is not None
    assert meta["title"] == "Grobid Extracted Title"


@respx.mock
def test_extract_non_200_returns_none(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    respx.post("http://grobid.test/api/processHeaderDocument").mock(
        return_value=httpx.Response(500, text="server error")
    )
    assert GrobidClient("http://grobid.test").extract(pdf) is None


@respx.mock
def test_extract_network_error_returns_none(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    respx.post("http://grobid.test/api/processHeaderDocument").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    assert GrobidClient("http://grobid.test").extract(pdf) is None


@respx.mock
def test_extract_timeout_returns_none(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    respx.post("http://grobid.test/api/processHeaderDocument").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    assert GrobidClient("http://grobid.test").extract(pdf) is None
