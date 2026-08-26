from __future__ import annotations

from scholar_rag.core.types import AnnotatedBlock, AnnotatedDocument, MetadataResult
from scholar_rag.ingest.chunk import stage_chunks
from scholar_rag.ingest.staging import ChunkConfig


def _meta(**overrides: object) -> MetadataResult:
    values: dict[str, object] = {
        "title": "Synthetic Paper",
        "year": 2024,
        "journal": "Synth Journal",
        "doi": None,
        "authors": ["First Author", "Second Author"],
        "abstract": None,
        "source": "local",
        "keywords": (),
    }
    values.update(overrides)
    return MetadataResult(
        title=str(values["title"]),
        year=values["year"] if values["year"] is None else int(values["year"]),
        journal=values["journal"] if values["journal"] is None else str(values["journal"]),
        doi=values["doi"] if values["doi"] is None else str(values["doi"]),
        authors=list(values["authors"]) if isinstance(values["authors"], list) else list(str(a) for a in values["authors"]),
        abstract=values["abstract"] if values["abstract"] is None else str(values["abstract"]),
        source=str(values["source"]),
        keywords=tuple(str(k) for k in values["keywords"]),
    )


def _block(
    text: str,
    *,
    heading: bool = False,
    level: int = 0,
    section: str = "Other",
    confidence: float = 1.0,
) -> AnnotatedBlock:
    return AnnotatedBlock(
        text=text,
        is_heading=heading,
        heading_level=level,
        section=section,
        confidence=confidence,
    )


def _doc(*blocks: AnnotatedBlock) -> AnnotatedDocument:
    return AnnotatedDocument(blocks=tuple(blocks), sections=())


def _body(text: str, section: str = "Methods") -> AnnotatedBlock:
    return _block(text, section=section)


def test_empty_document_yields_no_chunks() -> None:
    assert stage_chunks(_doc(), _meta(), ChunkConfig()) == []


def test_headings_are_not_chunked_as_content() -> None:
    doc = _doc(
        _block("Methods", heading=True, level=2, section="Methods"),
        _body("A results observation", section="Methods"),
    )
    chunks = stage_chunks(doc, _meta(), ChunkConfig(min_chunk=10, max_chunk=100, overlap=0))
    assert len(chunks) == 1
    assert "Methods" not in chunks[0].content


def test_chunk_index_increments_from_zero() -> None:
    paragraphs = [_body(f"para number {i} " * 8, section="Methods") for i in range(40)]
    doc = _doc(*paragraphs)
    cfg = ChunkConfig(min_chunk=200, max_chunk=300, overlap=20)
    chunks = stage_chunks(doc, _meta(), cfg)
    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_index == 0


def test_heading_path_reflects_heading_nesting() -> None:
    doc = _doc(
        _block("Methods", heading=True, level=2, section="Methods"),
        _block("Model Architecture", heading=True, level=3, section="Methods"),
        _body("The model stacks two encoders and an attention head. " * 6, section="Methods"),
    )
    cfg = ChunkConfig(min_chunk=100, max_chunk=500, overlap=0)
    chunks = stage_chunks(doc, _meta(), cfg)
    assert chunks and chunks[0].heading_path == "Methods > Model Architecture"
    assert chunks[0].section == "Methods"


def test_chunk_lengths_respect_min_and_max() -> None:
    paragraphs = [_body(f"segment {'x' * 190} ", section="Results") for _ in range(12)]
    doc = _doc(*paragraphs)
    cfg = ChunkConfig(min_chunk=100, max_chunk=400, overlap=0)
    chunks = stage_chunks(doc, _meta(), cfg)
    assert chunks
    for chunk in chunks:
        assert len(chunk.content) >= cfg.min_chunk
        assert len(chunk.content) <= cfg.max_chunk


def test_large_paragraph_split_applies_overlap() -> None:
    filler = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu " * 30
    doc = _doc(_body(filler, section="Methods"))
    cfg = ChunkConfig(min_chunk=20, max_chunk=280, overlap=70)
    chunks = stage_chunks(doc, _meta(), cfg)
    assert len(chunks) >= 2
    assert chunks[1].content.startswith(chunks[0].content[-70:].strip())


def test_blocks_grouped_by_section_are_chunked_separately() -> None:
    doc = _doc(
        _body("methods text " * 12, section="Methods"),
        _body("results text " * 12, section="Results"),
    )
    cfg = ChunkConfig(min_chunk=20, max_chunk=2000, overlap=0)
    chunks = stage_chunks(doc, _meta(), cfg)
    sections = {c.section for c in chunks}
    assert sections == {"Methods", "Results"}


def test_metadata_carries_section_and_heading_path() -> None:
    doc = _doc(
        _block("Methods", heading=True, level=2, section="Methods"),
        _body("measurement protocol " * 12, section="Methods"),
    )
    cfg = ChunkConfig(min_chunk=20, max_chunk=2000, overlap=0)
    chunks = stage_chunks(doc, _meta(), cfg)
    assert chunks[0].metadata["section"] == "Methods"
    assert chunks[0].metadata["heading_path"] == "Methods"


def test_document_metadata_is_injected_into_every_chunk() -> None:
    doc = _doc(
        _block("Methods", heading=True, level=2, section="Methods"),
        _body("protocol details " * 6, section="Methods"),
    )
    cfg = ChunkConfig(min_chunk=20, max_chunk=2000, overlap=0)
    chunks = stage_chunks(doc, _meta(title="Unique Title", year=2023), cfg)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["title"] == "Unique Title"
        assert chunk.metadata["year"] == 2023
        assert chunk.metadata["first_author"] == "First Author"
        assert chunk.metadata["author_count"] == 2
