from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"


class BlockRole(StrEnum):
    UNKNOWN = "unknown"
    TITLE = "title"
    FRONT_MATTER = "front_matter"
    SECTION_HEADING = "section_heading"
    SUBSECTION_HEADING = "subsection_heading"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    BODY_PARAGRAPH = "body_paragraph"
    REFERENCE_ENTRY = "reference_entry"
    ACK_META = "ack_meta"
    SUMMARY_BOX = "summary_box"
    SUPPLEMENTARY_META = "supplementary_meta"


@dataclass(frozen=True)
class BlockSignals:
    starts_with_figure_token: bool = False
    starts_with_table_token: bool = False
    has_affiliation_keywords: bool = False
    has_received_dates: bool = False
    has_reference_markers: bool = False
    looks_like_short_label: bool = False
    looks_like_summary_heading: bool = False
    looks_like_caption_body: bool = False


@dataclass(frozen=True)
class DocumentBlock:
    index: int
    raw_text: str
    text: str
    kind: BlockKind
    heading_level: int = 0
    line_count: int = 1
    char_length: int = 0
    position_ratio: float = 0.0
    anchor_heading_index: int | None = None
    signals: BlockSignals = field(default_factory=BlockSignals)


@dataclass(frozen=True)
class StructuredDocument:
    source_text: str
    blocks: list[DocumentBlock]

    def heading_blocks(self) -> list[DocumentBlock]:
        return [block for block in self.blocks if block.kind == BlockKind.HEADING]

    def paragraph_blocks(self) -> list[DocumentBlock]:
        return [block for block in self.blocks if block.kind == BlockKind.PARAGRAPH]


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")
_FIGURE_TOKEN_RE = re.compile(r"^(?:fig(?:ure)?\.?)\s*[A-Za-z]?\d+", re.IGNORECASE)
_TABLE_TOKEN_RE = re.compile(r"^(?:table)\s*[A-Za-z]?\d+", re.IGNORECASE)
_REFERENCE_MARKER_RE = re.compile(r"^(?:\(?\d+\)?[\].:]?\s+|[A-Z][A-Za-z' -]+,\s*[A-Z]\.)")
_AFFILIATION_RE = re.compile(
    r"\b(department|university|institute|school|hospital|laboratory|centre|center)\b",
    re.IGNORECASE,
)
_DATE_META_RE = re.compile(
    r"\b(received|accepted|published|editorial decision|revised)\b",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(r"\b(summary|research article summary|significance)\b", re.IGNORECASE)
_CAPTION_BODY_RE = re.compile(
    r"\b(shown from top to bottom|scale bar|cells were treated|left to right)\b",
    re.IGNORECASE,
)


def extract_document_blocks(text: str) -> StructuredDocument:
    if not text.strip():
        return StructuredDocument(source_text=text, blocks=[])

    raw_blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    total = max(len(raw_blocks) - 1, 1)
    blocks: list[DocumentBlock] = []
    current_heading_index: int | None = None

    for index, raw_block in enumerate(raw_blocks):
        match = _HEADING_RE.match(raw_block)
        if match:
            kind = BlockKind.HEADING
            heading_level = len(match.group(1))
            normalized = match.group(2).strip()
        else:
            kind = BlockKind.PARAGRAPH
            heading_level = 0
            normalized = raw_block

        signals = _detect_block_signals(normalized)
        block = DocumentBlock(
            index=index,
            raw_text=raw_block,
            text=normalized,
            kind=kind,
            heading_level=heading_level,
            line_count=len([line for line in raw_block.splitlines() if line.strip()]),
            char_length=len(normalized),
            position_ratio=index / total,
            anchor_heading_index=current_heading_index,
            signals=signals,
        )
        blocks.append(block)

        if kind == BlockKind.HEADING:
            current_heading_index = index

    return StructuredDocument(source_text=text, blocks=blocks)


def _detect_block_signals(text: str) -> BlockSignals:
    normalized = text.strip()
    lowered = normalized.lower()
    compact = normalized.replace(" ", "")

    return BlockSignals(
        starts_with_figure_token=bool(_FIGURE_TOKEN_RE.match(normalized)),
        starts_with_table_token=bool(_TABLE_TOKEN_RE.match(normalized)),
        has_affiliation_keywords=bool(_AFFILIATION_RE.search(normalized)),
        has_received_dates=bool(_DATE_META_RE.search(normalized)),
        has_reference_markers=bool(_REFERENCE_MARKER_RE.match(normalized)),
        looks_like_short_label=len(compact) <= 12
        and bool(re.fullmatch(r"[A-Za-z0-9_.:-]+", compact)),
        looks_like_summary_heading=bool(_SUMMARY_RE.search(normalized)),
        looks_like_caption_body=bool(_CAPTION_BODY_RE.search(lowered)),
    )
