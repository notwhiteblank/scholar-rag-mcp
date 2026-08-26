from __future__ import annotations

import re
from dataclasses import dataclass

from scholar_rag.ingest.annotate.document_structure import (
    BlockKind,
    BlockRole,
    DocumentBlock,
    StructuredDocument,
)

_CANONICAL_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "results",
    "discussion",
    "results and discussion",
    "methods",
    "materials and methods",
    "methodology",
    "experimental procedures",
    "experimental section",
    "conclusion",
    "conclusions",
    "references",
    "references and notes",
    "bibliography",
}

_ACK_META_HEADINGS = {
    "acknowledgments",
    "acknowledgements",
    "funding",
    "author contributions",
    "competing interests",
    "conflict of interest",
    "data availability",
    "data and materials availability",
    "code availability",
}

_SUPPLEMENTARY_HEADINGS = {
    "supplementary material",
    "supplementary materials",
    "supplementary data",
    "supplementary information",
    "supporting information",
    "extended data",
}

_REFERENCE_HEADINGS = {
    "references",
    "references and notes",
    "bibliography",
}

_FRONT_MATTER_HEADINGS = {
    "these authors contributed equally to this work.",
}

_INLINE_SECTION_LABEL_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)[.)]?\s+"
    r"(abstract|introduction|background|results?|discussion|methods?|materials and methods|conclusions?)\s*$",
    re.IGNORECASE,
)
_SPACED_LABEL_RE = re.compile(r"^(?:[A-Z]\s+){3,}[A-Z]$")
_KEYWORDS_RE = re.compile(r"^(?:key\s*words?|keywords)\s*:", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_CORRESPONDENCE_RE = re.compile(r"^(?:correspondence|corresponding author)\s*:", re.IGNORECASE)
_JOURNAL_META_RE = re.compile(
    r"\b("
    r"contents lists available|journal homepage|sciencedirect|elsevier|"
    r"check for updates|available online"
    r")\b",
    re.IGNORECASE,
)
_AUTHOR_NAME_RE = re.compile(
    r"[A-Z][A-Za-z'`-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'`-]+)+(?:\d+|\*)?"
)


@dataclass(frozen=True)
class RecoveredBlock:
    block: DocumentBlock
    role: BlockRole
    confidence: float
    rationale: str

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.7


@dataclass(frozen=True)
class StructureRecoveryResult:
    document: StructuredDocument
    blocks: list[RecoveredBlock]

    def role_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.role.value] = counts.get(block.role.value, 0) + 1
        return counts

    def low_confidence_indices(self) -> list[int]:
        return [block.block.index for block in self.blocks if block.is_low_confidence]

    def get(self, block_index: int) -> RecoveredBlock:
        return self.blocks[block_index]


class BaseStructureRecoverer:
    def recover(self, document: StructuredDocument) -> StructureRecoveryResult:
        raise NotImplementedError


class RuleBasedStructureRecoverer(BaseStructureRecoverer):
    def recover(self, document: StructuredDocument) -> StructureRecoveryResult:
        recovered: list[RecoveredBlock] = []
        active_major_heading_role = BlockRole.UNKNOWN
        first_structural_marker_index = _find_first_structural_marker_index(document)

        for block in document.blocks:
            result = self._recover_block(
                block,
                active_major_heading_role,
                document,
                first_structural_marker_index,
            )
            recovered.append(result)
            if result.role in {
                BlockRole.SECTION_HEADING,
                BlockRole.ACK_META,
                BlockRole.SUPPLEMENTARY_META,
                BlockRole.SUMMARY_BOX,
            }:
                active_major_heading_role = result.role

        return StructureRecoveryResult(document=document, blocks=recovered)

    def _recover_block(
        self,
        block: DocumentBlock,
        active_major_heading_role: BlockRole,
        document: StructuredDocument,
        first_structural_marker_index: int,
    ) -> RecoveredBlock:
        if block.kind == BlockKind.HEADING:
            return self._recover_heading(block, active_major_heading_role, document)
        return self._recover_paragraph(
            block,
            active_major_heading_role,
            document,
            first_structural_marker_index,
        )

    def _recover_heading(
        self,
        block: DocumentBlock,
        active_major_heading_role: BlockRole,
        document: StructuredDocument,
    ) -> RecoveredBlock:
        normalized = _normalize_heading(block.text)

        if block.signals.starts_with_figure_token:
            return RecoveredBlock(block, BlockRole.FIGURE_CAPTION, 0.99, "figure-token heading")

        if block.signals.starts_with_table_token:
            return RecoveredBlock(block, BlockRole.TABLE_CAPTION, 0.99, "table-token heading")

        if normalized in _FRONT_MATTER_HEADINGS:
            return RecoveredBlock(block, BlockRole.FRONT_MATTER, 0.96, "known front-matter heading")

        if normalized in _ACK_META_HEADINGS:
            return RecoveredBlock(block, BlockRole.ACK_META, 0.96, "ack/meta heading")

        if normalized in _SUPPLEMENTARY_HEADINGS:
            return RecoveredBlock(block, BlockRole.SUPPLEMENTARY_META, 0.96, "supplementary heading")

        if block.signals.looks_like_summary_heading:
            return RecoveredBlock(block, BlockRole.SUMMARY_BOX, 0.92, "summary-like heading")

        if normalized in _CANONICAL_SECTION_HEADINGS:
            role = (
                BlockRole.SECTION_HEADING
                if block.heading_level <= 1
                else BlockRole.SUBSECTION_HEADING
            )
            rationale = (
                "canonical section heading"
                if role == BlockRole.SECTION_HEADING
                else "canonical subsection heading"
            )
            return RecoveredBlock(block, role, 0.95, rationale)

        if (
            block.index == 0
            and block.position_ratio <= 0.08
            and (block.char_length >= 20 or _has_preamble_context(document, block.index))
        ):
            return RecoveredBlock(block, BlockRole.TITLE, 0.97, "leading long heading")

        if active_major_heading_role in {
            BlockRole.SECTION_HEADING,
            BlockRole.SUMMARY_BOX,
            BlockRole.SUPPLEMENTARY_META,
        }:
            return RecoveredBlock(
                block, BlockRole.SUBSECTION_HEADING, 0.78, "nested under active major heading"
            )

        if 0.05 <= block.position_ratio <= 0.92:
            return RecoveredBlock(
                block, BlockRole.SUBSECTION_HEADING, 0.62, "mid-document free-form heading"
            )

        return RecoveredBlock(block, BlockRole.UNKNOWN, 0.35, "unresolved heading")

    def _recover_paragraph(
        self,
        block: DocumentBlock,
        active_major_heading_role: BlockRole,
        document: StructuredDocument,
        first_structural_marker_index: int,
    ) -> RecoveredBlock:
        if block.signals.looks_like_caption_body:
            return RecoveredBlock(block, BlockRole.FIGURE_CAPTION, 0.92, "caption-like paragraph")

        if block.signals.starts_with_table_token:
            return RecoveredBlock(block, BlockRole.TABLE_CAPTION, 0.95, "table-token paragraph")

        if block.signals.has_reference_markers and block.position_ratio >= 0.6:
            return RecoveredBlock(
                block, BlockRole.REFERENCE_ENTRY, 0.86, "reference-like tail paragraph"
            )

        if active_major_heading_role == BlockRole.ACK_META:
            return RecoveredBlock(block, BlockRole.ACK_META, 0.95, "inside ack/meta region")

        if active_major_heading_role == BlockRole.SUPPLEMENTARY_META:
            return RecoveredBlock(
                block, BlockRole.SUPPLEMENTARY_META, 0.94, "inside supplementary region"
            )

        if (
            active_major_heading_role == BlockRole.UNKNOWN
            and block.index <= first_structural_marker_index
        ):
            if _looks_like_summary_label_paragraph(block.text):
                return RecoveredBlock(
                    block, BlockRole.SUMMARY_BOX, 0.91, "summary-like paragraph label"
                )

            if (
                block.signals.has_affiliation_keywords
                or block.signals.has_received_dates
                or _looks_like_preamble_metadata(block.text)
            ):
                return RecoveredBlock(
                    block, BlockRole.FRONT_MATTER, 0.9, "preamble metadata paragraph"
                )

            if _looks_like_preamble_title(block, document, first_structural_marker_index):
                return RecoveredBlock(block, BlockRole.TITLE, 0.91, "title-like preamble paragraph")

        if active_major_heading_role == BlockRole.SUMMARY_BOX and _looks_like_preamble_metadata(
            block.text
        ):
            return RecoveredBlock(
                block, BlockRole.FRONT_MATTER, 0.9, "summary-region metadata paragraph"
            )

        if (
            active_major_heading_role == BlockRole.UNKNOWN
            and (block.signals.has_affiliation_keywords or block.signals.has_received_dates)
            and (block.position_ratio <= 0.35 or block.anchor_heading_index == 0)
        ):
            return RecoveredBlock(
                block, BlockRole.FRONT_MATTER, 0.93, "front-matter metadata paragraph"
            )

        if block.signals.looks_like_short_label:
            return RecoveredBlock(block, BlockRole.UNKNOWN, 0.42, "short unresolved label")

        return RecoveredBlock(block, BlockRole.BODY_PARAGRAPH, 0.88, "default body paragraph")


def recover_document_structure(
    document: StructuredDocument,
    recoverer: BaseStructureRecoverer | None = None,
) -> StructureRecoveryResult:
    effective_recoverer = recoverer or RuleBasedStructureRecoverer()
    return effective_recoverer.recover(document)


def _normalize_heading(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _find_first_structural_marker_index(document: StructuredDocument) -> int:
    for block in document.blocks:
        normalized = _normalize_heading(block.text)
        if block.kind == BlockKind.HEADING and (
            normalized in _CANONICAL_SECTION_HEADINGS
            or normalized in _ACK_META_HEADINGS
            or normalized in _SUPPLEMENTARY_HEADINGS
        ):
            return block.index
        if block.kind == BlockKind.PARAGRAPH and (
            _INLINE_SECTION_LABEL_RE.match(block.text)
            or _looks_like_summary_label_paragraph(block.text)
        ):
            return block.index
    return len(document.blocks)


def _looks_like_summary_label_paragraph(text: str) -> bool:
    normalized = _normalize_heading(_collapse_spaced_caps(text))
    return normalized in {"abstract", "summary", "article info"}


def _collapse_spaced_caps(text: str) -> str:
    stripped = " ".join(text.strip().split())
    if _SPACED_LABEL_RE.fullmatch(stripped):
        return stripped.replace(" ", "")
    return stripped


def _looks_like_preamble_metadata(text: str) -> bool:
    stripped = " ".join(text.strip().split())
    if not stripped:
        return False
    if _KEYWORDS_RE.match(stripped):
        return True
    if _CORRESPONDENCE_RE.match(stripped) or _EMAIL_RE.search(stripped):
        return True
    if _JOURNAL_META_RE.search(stripped):
        return True
    return bool(_looks_like_author_list(stripped))


def _looks_like_author_list(text: str) -> bool:
    matches = _AUTHOR_NAME_RE.findall(text)
    if len(matches) >= 2:
        return True
    return len(matches) == 1 and ("*" in text or any(ch.isdigit() for ch in text))


def _looks_like_preamble_title(
    block: DocumentBlock,
    document: StructuredDocument,
    first_structural_marker_index: int,
) -> bool:
    text = " ".join(block.text.strip().split())
    if not text:
        return False
    if block.index >= min(first_structural_marker_index, 12):
        return False
    if not (40 <= block.char_length <= 260):
        return False
    if _looks_like_preamble_metadata(text):
        return False
    if text[-1] in ".;!?":
        return False
    if _looks_like_summary_label_paragraph(text):
        return False
    if sum(ch in ".;:!?" for ch in text) > 1:
        return False

    return _has_preamble_context(document, block.index)


def _has_preamble_context(document: StructuredDocument, index: int) -> bool:
    nearby = document.blocks[index + 1 : index + 5]
    return any(
        candidate.signals.has_affiliation_keywords
        or candidate.signals.has_received_dates
        or _looks_like_preamble_metadata(candidate.text)
        for candidate in nearby
    )
