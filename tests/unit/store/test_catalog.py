import os

import pytest

from scholar_rag.core.types import MetadataResult
from scholar_rag.store.catalog import Catalog


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def catalog(tmp_path):
    return Catalog(tmp_path / "catalog.sqlite3")


def make_meta(
    title: str,
    authors=(),
    year=None,
    journal=None,
    doi=None,
    abstract=None,
    keywords=(),
) -> MetadataResult:
    return MetadataResult(
        title=title,
        year=year,
        journal=journal,
        doi=doi,
        authors=list(authors),
        abstract=abstract,
        keywords=tuple(keywords),
        source="local",
    )


def make_doc(
    catalog: Catalog,
    doc_id: str,
    title: str,
    *,
    authors=(),
    year=None,
    journal=None,
    doi=None,
    abstract=None,
    keywords=(),
    source_path="",
    content_hash=None,
) -> None:
    catalog.upsert_document(
        make_meta(title, authors, year, journal, doi, abstract, keywords),
        doc_id,
        source_path,
        content_hash or doc_id,
    )


def test_upsert_and_get_document(catalog):
    make_doc(
        catalog,
        "doc1",
        "Attention Networks",
        authors=["A", "B", "C"],
        year=2022,
        journal="JMLR",
        doi="10.1/2",
        abstract="Abstract text",
        keywords=("attention", "network"),
        source_path="/src/1.pdf",
    )
    row = catalog.get_document("doc1")
    assert row is not None
    assert row["doc_id"] == "doc1"
    assert row["title"] == "Attention Networks"
    assert row["year"] == 2022
    assert row["journal"] == "JMLR"
    assert row["doi"] == "10.1/2"
    assert row["abstract"] == "Abstract text"
    assert row["first_author"] == "A"
    assert row["author_count"] == 3
    assert row["source_path"] == "/src/1.pdf"
    assert row["authors"] == ["A", "B", "C"]
    assert row["keywords"] == ["attention", "network"]
    assert row["content_hash"] == "doc1"
    assert row["parse_status"] == "done"
    assert row["added_at"]


def test_upsert_empty_authors_uses_empty_first_author(catalog):
    make_doc(catalog, "d", "Solo")
    row = catalog.get_document("d")
    assert row["first_author"] == ""
    assert row["author_count"] == 0


def test_get_document_missing_returns_none(catalog):
    assert catalog.get_document("missing") is None


def test_upsert_same_doc_id_is_idempotent(catalog):
    make_doc(catalog, "doc1", "V1", authors=["X"], keywords=("old",), content_hash="h1")
    make_doc(catalog, "doc1", "V2", authors=["Y", "Z"], keywords=("new",), content_hash="h2")
    assert catalog.count_documents() == 1
    row = catalog.get_document("doc1")
    assert row["title"] == "V2"
    assert row["authors"] == ["Y", "Z"]
    assert row["keywords"] == ["new"]
    assert row["content_hash"] == "h2"


def test_count_documents(catalog):
    make_doc(catalog, "a", "A")
    make_doc(catalog, "b", "B")
    make_doc(catalog, "c", "C")
    assert catalog.count_documents() == 3


def test_insert_chunks_and_cascade_delete(catalog):
    make_doc(catalog, "d", "T")
    catalog.insert_chunks("d", ["c1", "c2", "c3"])
    catalog.delete_document("d")
    assert catalog.count_documents() == 0
    assert catalog.get_document("d") is None


def test_delete_document_missing_is_idempotent(catalog):
    make_doc(catalog, "d", "T")
    catalog.delete_document("nope")
    assert catalog.count_documents() == 1
    catalog.delete_document("d")
    catalog.delete_document("d")
    assert catalog.count_documents() == 0


def test_list_documents_sorts_and_paginates(catalog):
    make_doc(catalog, "d1", "Beta", year=2020)
    make_doc(catalog, "d2", "Alpha", year=2022)
    make_doc(catalog, "d3", "Gamma", year=2021)
    assert catalog.count_documents() == 3

    total, rows = catalog.list_documents("added", 1, 10)
    assert total == 3
    assert len(rows) == 3

    total, rows = catalog.list_documents("title", 1, 10)
    assert [r["title"] for r in rows] == ["Alpha", "Beta", "Gamma"]

    total, rows = catalog.list_documents("year", 1, 10)
    assert [r["year"] for r in rows] == [2020, 2021, 2022]

    _, page1 = catalog.list_documents("added", 1, 2)
    assert len(page1) == 2
    _, page2 = catalog.list_documents("added", 2, 2)
    assert len(page2) == 1
    assert not {r["doc_id"] for r in page1} & {r["doc_id"] for r in page2}
    for row in page1 + page2:
        assert set(row) == {"doc_id", "title", "year", "journal", "first_author", "doi", "added_at"}


def test_list_documents_invalid_sort_raises(catalog):
    make_doc(catalog, "d", "T")
    with pytest.raises(ValueError):
        catalog.list_documents("bogus", 1, 10)


def test_search_by_fts_chinese_and_english_mixed(catalog):
    make_doc(
        catalog,
        "c1",
        "基于注意力机制的场景图检索",
        abstract="Attention based scene graph retrieval for robotics",
        keywords=("attention", "scene", "graph"),
        content_hash="c1",
    )
    make_doc(
        catalog,
        "c2",
        "Transformer 在文本摘要中的应用",
        abstract="A transformer based text summarization system",
        keywords=("transformer", "summarization"),
        content_hash="c2",
    )
    total, rows = catalog.search_documents("注意力", None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "c1"

    total, rows = catalog.search_documents("文本摘要", None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "c2"

    total, rows = catalog.search_documents("scene graph", None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "c1"

    total, rows = catalog.search_documents("summarization", None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "c2"


def test_search_fts_escapes_special_characters(catalog):
    make_doc(catalog, "d1", "Attention Networks", abstract="Deep learning survey", content_hash="d1")
    make_doc(catalog, "d2", "GPT Models", abstract="Language model scaling", content_hash="d2")
    total, rows = catalog.search_documents('"gpt" models', None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "d2"
    total, rows = catalog.search_documents("attention survey", None, None, None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "d1"
    total, _ = catalog.search_documents("gpt survey", None, None, None, None, None, 1, 10)
    assert total == 0


def test_search_combined_filters(catalog):
    make_doc(catalog, "d1", "Graph Nets", authors=["Aaaa"], year=2020, journal="TMLR", doi="x", content_hash="d1")
    make_doc(catalog, "d2", "Graph Mining", authors=["Bbbb"], year=2021, journal="TKDE", doi="y", content_hash="d2")
    make_doc(catalog, "d3", "Other", authors=["Cccc"], year=2021, journal="TMLR", doi="z", content_hash="d3")

    total, rows = catalog.search_documents(None, "Graph", None, None, None, None, 1, 10)
    assert total == 2

    total, rows = catalog.search_documents(None, None, "Aaaa", None, None, None, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "d1"

    total, rows = catalog.search_documents(None, None, "a", None, None, None, 1, 10)
    assert total == 1

    total, rows = catalog.search_documents(None, None, None, "TMLR", None, None, 1, 10)
    assert total == 2

    total, rows = catalog.search_documents(None, None, None, None, 2021, 2021, 1, 10)
    assert total == 2

    total, rows = catalog.search_documents("Graph", None, None, None, 2021, 2021, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "d2"

    total, rows = catalog.search_documents(None, None, None, None, None, None, 1, 10)
    assert total == 3


def test_search_pagination(catalog):
    for i in range(5):
        make_doc(catalog, f"d{i}", f"Doc {i}", content_hash=f"d{i}")
    total, page1 = catalog.search_documents(None, None, None, None, None, None, 1, 2)
    assert total == 5
    assert len(page1) == 2
    total, page3 = catalog.search_documents(None, None, None, None, None, None, 3, 2)
    assert len(page3) == 1


def test_search_malformed_fts_query_treated_as_no_fts_filter(catalog):
    """A whitespace/quote-only FTS query is treated as an empty FTS filter."""
    make_doc(catalog, "d1", "Attention", year=2021, content_hash="d1")
    make_doc(catalog, "d2", "Models", year=2022, content_hash="d2")
    for meaningless in ("   ", '""', '"'):
        total, rows = catalog.search_documents(meaningless, None, None, None, None, None, 1, 10)
        assert total == 2
    total, rows = catalog.search_documents("   ", None, None, None, 2021, 2021, 1, 10)
    assert total == 1
    assert rows[0]["doc_id"] == "d1"
