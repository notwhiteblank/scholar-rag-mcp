from __future__ import annotations

from scholar_rag.core.types import AnnotatedDocument, Chunk, MetadataResult
from scholar_rag.ingest.staging import ChunkConfig

_SENTENCE_SEPARATORS = (". ", "? ", "! ", ".\n", "?\n", "!\n")


def stage_chunks(
    annotated: AnnotatedDocument,
    meta: MetadataResult,
    cfg: ChunkConfig,
) -> list[Chunk]:
    groups = _collect_body_groups(annotated)
    chunks: list[Chunk] = []
    index = 0
    for section, heading_path, paragraphs in groups:
        text = "\n\n".join(paragraphs)
        metadata = _chunk_metadata(meta, section, heading_path)
        for content in _split_text(text, cfg):
            chunks.append(
                Chunk(
                    chunk_index=index,
                    content=content,
                    section=section,
                    heading_path=heading_path,
                    metadata=dict(metadata),
                )
            )
            index += 1
    return chunks


def _collect_body_groups(
    annotated: AnnotatedDocument,
) -> list[tuple[str, str, list[str]]]:
    path_stack: list[tuple[str, int]] = []
    groups: list[tuple[str, str, list[str]]] = []
    for block in annotated.blocks:
        if block.is_heading:
            while path_stack and path_stack[-1][1] >= block.heading_level:
                path_stack.pop()
            path_stack.append((block.text, block.heading_level))
            continue
        heading_path = " > ".join(text for text, _level in path_stack)
        key = (block.section, heading_path)
        if groups and groups[-1][0] == key[0] and groups[-1][1] == key[1]:
            groups[-1][2].append(block.text)
        else:
            groups.append((block.section, heading_path, [block.text]))
    return groups


def _chunk_metadata(meta: MetadataResult, section: str, heading_path: str) -> dict[str, object]:
    return {
        "title": meta.title,
        "year": meta.year,
        "journal": meta.journal,
        "doi": meta.doi,
        "first_author": meta.authors[0] if meta.authors else "",
        "author_count": len(meta.authors),
        "section": section,
        "heading_path": heading_path,
    }


def _split_text(text: str, cfg: ChunkConfig) -> list[str]:
    if not text.strip():
        return []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    raw_chunks: list[str] = []
    current: str = ""
    for paragraph in paragraphs:
        if len(paragraph) > cfg.max_chunk:
            if current:
                raw_chunks.append(current)
                current = ""
            raw_chunks.extend(_split_large_paragraph(paragraph, cfg))
            continue
        potential = f"{current}\n\n{paragraph}" if current else paragraph
        if len(potential) <= cfg.max_chunk:
            current = potential
        else:
            if current:
                raw_chunks.append(current)
            current = paragraph
    if current:
        raw_chunks.append(current)
    return _merge_small_chunks(raw_chunks, cfg)


def _split_large_paragraph(text: str, cfg: ChunkConfig) -> list[str]:
    if len(text) <= cfg.max_chunk:
        return [text]
    chunks: list[str] = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + cfg.max_chunk, len(text))
        if end_pos < len(text):
            candidate = text[current_pos:end_pos]
            best_break = -1
            for separator in _SENTENCE_SEPARATORS:
                position = candidate.rfind(separator)
                if position > len(candidate) // 2:
                    best_break = max(best_break, position + len(separator))
            if best_break > 0:
                end_pos = current_pos + best_break
        chunks.append(text[current_pos:end_pos].strip())
        current_pos = end_pos - cfg.overlap if end_pos < len(text) else end_pos
    return [chunk for chunk in chunks if chunk]


def _merge_small_chunks(raw_chunks: list[str], cfg: ChunkConfig) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(raw_chunks):
        chunk = raw_chunks[index]
        while len(chunk) < cfg.min_chunk and index + 1 < len(raw_chunks):
            index += 1
            chunk = f"{chunk}\n\n{raw_chunks[index]}"
        merged.append(chunk)
        index += 1
    return merged
