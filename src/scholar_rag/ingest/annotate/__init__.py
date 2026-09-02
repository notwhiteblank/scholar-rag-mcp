from __future__ import annotations

from pathlib import Path

from scholar_rag.core.config import Settings
from scholar_rag.core.types import AnnotatedBlock, AnnotatedDocument
from scholar_rag.ingest.annotate.consistency_repair import (
    ParsedParagraph,
    RepairResult,
    repair_paragraphs,
)
from scholar_rag.ingest.annotate.document_structure import (
    BlockKind,
    BlockRole,
    extract_document_blocks,
)
from scholar_rag.ingest.annotate.resolver import SectionResolver
from scholar_rag.ingest.annotate.section_decode import (
    DecodedBlock,
    SectionDecodeResult,
    decode_document_sections,
)
from scholar_rag.ingest.annotate.structure_recovery import recover_document_structure
from scholar_rag.models import get_chat_client

__all__ = ["stage_annotate"]


def stage_annotate(cleaned: str) -> AnnotatedDocument:
    structured = extract_document_blocks(cleaned)
    recovered = recover_document_structure(structured)
    decoded = decode_document_sections(recovered)
    settings = Settings.load()
    if settings.annotation_resolver_enabled and _chat_configured(settings):
        decoded = _run_resolver(decoded, settings)
    paragraphs = decode_to_paragraphs(decoded)
    repaired = repair_paragraphs(
        paragraphs,
        low_confidence_indices=recovered.low_confidence_indices(),
    )
    blocks = build_blocks(decoded, repaired)
    sections = build_sections(blocks)
    return AnnotatedDocument(blocks=blocks, sections=sections)


def _chat_configured(settings: Settings) -> bool:
    return bool(settings.chat_base_url)


def _resolver_cache_dir(settings: Settings) -> Path:
    return settings.data_dir / "cache" / "resolver"


def _run_resolver(decoded: SectionDecodeResult, settings: Settings) -> SectionDecodeResult:
    try:
        chat_client = get_chat_client()
    except Exception:
        return decoded
    try:
        resolver = SectionResolver()
        return resolver.resolve(
            decoded,
            chat_client=chat_client,
            cache_dir=_resolver_cache_dir(settings),
        )
    except Exception:
        return decoded


def decode_to_paragraphs(decoded: SectionDecodeResult) -> list[ParsedParagraph]:
    paragraphs: list[ParsedParagraph] = []
    for item in decoded.blocks:
        if item.recovered_block.block.kind != BlockKind.PARAGRAPH:
            continue
        paragraphs.append(
            ParsedParagraph(
                text=item.recovered_block.block.text,
                section=item.section,
                subsection=item.subsection,
                content_type=_role_to_content_type(item.recovered_block.role),
            )
        )
    return paragraphs


def build_blocks(decoded: SectionDecodeResult, repaired: RepairResult) -> tuple[AnnotatedBlock, ...]:
    paragraph_index = 0
    blocks: list[AnnotatedBlock] = []
    for item in decoded.blocks:
        block = item.recovered_block.block
        if block.kind == BlockKind.HEADING:
            blocks.append(_to_annotated_block(item))
            continue
        if paragraph_index >= len(repaired.paragraphs):
            continue
        paragraph = repaired.paragraphs[paragraph_index]
        paragraph_index += 1
        if paragraph.content_type != "body":
            continue
        blocks.append(_to_annotated_block(item))
    return tuple(blocks)


def build_sections(blocks: tuple[AnnotatedBlock, ...]) -> tuple[tuple[str, int], ...]:
    ordered: list[str] = []
    counts: dict[str, int] = {}
    for block in blocks:
        if block.section not in counts:
            ordered.append(block.section)
            counts[block.section] = 0
        if not block.is_heading:
            counts[block.section] += len(block.text)
    return tuple((name, counts[name]) for name in ordered)


def _to_annotated_block(item: DecodedBlock) -> AnnotatedBlock:
    block = item.recovered_block.block
    return AnnotatedBlock(
        text=block.text,
        is_heading=block.kind == BlockKind.HEADING,
        heading_level=block.heading_level if block.kind == BlockKind.HEADING else 0,
        section=item.section,
        confidence=item.confidence,
    )


def _role_to_content_type(role: BlockRole) -> str:
    if role == BlockRole.BODY_PARAGRAPH:
        return "body"
    if role == BlockRole.REFERENCE_ENTRY:
        return "reference"
    return "noise"
