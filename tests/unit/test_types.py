import dataclasses

import pytest

from scholar_rag.core.types import (
    AnnotatedBlock,
    AnnotatedDocument,
    Chunk,
    JobRecord,
    MetadataResult,
    RetrievedChunk,
)


def make_instances():
    return {
        "MetadataResult": (
            MetadataResult,
            dict(title="t", year=None, journal=None, doi=None, authors=["a"], abstract=None, source="local"),
        ),
        "AnnotatedBlock": (
            AnnotatedBlock,
            dict(text="text", is_heading=False, heading_level=0, section="Other", confidence=0.5),
        ),
        "AnnotatedDocument": (AnnotatedDocument, dict(blocks=(), sections=())),
        "Chunk": (Chunk, dict(chunk_index=0, content="c", section="s", heading_path="h", metadata={})),
        "RetrievedChunk": (
            RetrievedChunk,
            dict(content="c", metadata={}, embed_score=0.9, rerank_score=None),
        ),
        "JobRecord": (
            JobRecord,
            dict(
                job_id="j1",
                type="add_document",
                kb="kb",
                status="pending",
                done=0,
                total=1,
                current_doc=None,
                error=None,
                result=None,
                created_at="t",
                updated_at="t",
            ),
        ),
    }


@pytest.mark.parametrize("cls", list(make_instances()))
def test_dataclasses_are_frozen(cls):
    factory, kwargs = make_instances()[cls]
    instance = factory(**kwargs)
    field_name = dataclasses.fields(instance)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field_name, "x")


def test_metadata_result_keywords_default_empty():
    result = MetadataResult(title="t", year=None, journal=None, doi=None, authors=[], abstract=None, source="local")
    assert result.keywords == ()


def test_metadata_result_fields_are_set():
    result = MetadataResult(
        title="t",
        year=2024,
        journal="j",
        doi="10.1/x",
        authors=["a", "b"],
        abstract="abs",
        source="crossref",
        keywords=("k1", "k2"),
    )
    assert result.title == "t"
    assert result.year == 2024
    assert result.journal == "j"
    assert result.doi == "10.1/x"
    assert result.authors == ["a", "b"]
    assert result.abstract == "abs"
    assert result.source == "crossref"
    assert result.keywords == ("k1", "k2")


def test_retrieved_chunk_accepts_explicit_none_rerank_score():
    chunk = RetrievedChunk(content="c", metadata={}, embed_score=0.9, rerank_score=None)
    assert chunk.rerank_score is None


def test_retrieved_chunk_requires_rerank_score():
    with pytest.raises(TypeError):
        RetrievedChunk(content="c", metadata={}, embed_score=0.9)


def test_job_record_fields_are_set():
    record = JobRecord(
        job_id="j1",
        type="add_document",
        kb="kb",
        status="running",
        done=1,
        total=3,
        current_doc="doc.pdf",
        error=None,
        result={"ok": True},
        created_at="t0",
        updated_at="t1",
    )
    assert record.job_id == "j1"
    assert record.type == "add_document"
    assert record.kb == "kb"
    assert record.status == "running"
    assert record.done == 1
    assert record.total == 3
    assert record.current_doc == "doc.pdf"
    assert record.error is None
    assert record.result == {"ok": True}
    assert record.created_at == "t0"
    assert record.updated_at == "t1"


def test_metadata_result_requires_all_required_fields():
    with pytest.raises(TypeError):
        MetadataResult(title="t")
