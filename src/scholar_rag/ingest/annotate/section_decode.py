from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from scholar_rag.ingest.annotate.document_structure import BlockKind, BlockRole
from scholar_rag.ingest.annotate.structure_recovery import (
    RecoveredBlock,
    StructureRecoveryResult,
)

SECTION_OTHER = "Other"
SECTION_AUTHOR_INFO = "Author_Info"
SECTION_ACK = "Acknowledgments"
SECTION_SUPPLEMENTARY = "Supplementary"
SECTION_REFERENCES = "References"
SECTION_ABSTRACT = "Abstract"
SECTION_INTRO = "Introduction"
SECTION_RESULTS = "Results"
SECTION_DISCUSSION = "Discussion"
SECTION_METHODS = "Methods"
SECTION_CONCLUSION = "Conclusion"

_DIRECT_HEADING_SECTION_MAP = {
    "abstract": SECTION_ABSTRACT,
    "summary": SECTION_ABSTRACT,
    "graphical abstract": SECTION_ABSTRACT,
    "introduction": SECTION_INTRO,
    "background": SECTION_INTRO,
    "results": SECTION_RESULTS,
    "results and discussion": SECTION_RESULTS,
    "discussion": SECTION_DISCUSSION,
    "methods": SECTION_METHODS,
    "materials and methods": SECTION_METHODS,
    "methodology": SECTION_METHODS,
    "experimental procedures": SECTION_METHODS,
    "experimental section": SECTION_METHODS,
    "conclusion": SECTION_CONCLUSION,
    "conclusions": SECTION_CONCLUSION,
    "references": SECTION_REFERENCES,
    "references and notes": SECTION_REFERENCES,
    "bibliography": SECTION_REFERENCES,
    "supplementary material": SECTION_SUPPLEMENTARY,
    "supplementary materials": SECTION_SUPPLEMENTARY,
    "supplementary data": SECTION_SUPPLEMENTARY,
    "supplementary information": SECTION_SUPPLEMENTARY,
    "supporting information": SECTION_SUPPLEMENTARY,
    "acknowledgments": SECTION_ACK,
    "acknowledgements": SECTION_ACK,
    "funding": SECTION_ACK,
    "data and materials availability": SECTION_AUTHOR_INFO,
    "data availability": SECTION_AUTHOR_INFO,
    "code availability": SECTION_AUTHOR_INFO,
}

_METHODS_HEADING_CUES = (
    "method",
    "methods",
    "experimental",
    "procedure",
    "protocol",
    "assay",
    "simulation",
    "simulations",
    "analysis",
    "calculation",
    "calculations",
    "statistics",
    "statistical",
    "materials",
)
_RESULTS_HEADING_CUES = (
    "result",
    "results",
    "predict",
    "prediction",
    "identify",
    "identifies",
    "construction",
    "design",
    "characterization",
    "effect",
    "effects",
    "regulation",
    "regulates",
    "promotes",
    "reveals",
    "revealed",
    "find",
    "finds",
)
_DISCUSSION_HEADING_CUES = (
    "discussion",
    "model",
    "implication",
    "implications",
    "perspective",
    "perspectives",
    "outlook",
    "dynamics",
    "structure",
    "location",
    "composition",
    "function",
)
_INTRO_HEADING_CUES = (
    "overview",
    "background",
    "introduction",
    "context",
)

_PAGE_MARKER_RE = re.compile(r"^\d+\s+of\s+\d+$", re.IGNORECASE)
_RUNNING_HEADER_RE = re.compile(
    r"\b("
    r"science advances|j\.?\s*biol\.?\s*chem|asbmb|"
    r"cold spring harb|focus"
    r")\b",
    re.IGNORECASE,
)
_AUTHOR_NAME_RE = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+(?:\d+|\*)?)")
_RESULTS_BODY_RE = re.compile(
    r"\b("
    r"we (?:found|show|observed|identify|identified|sought)|"
    r"our results|these results|this study (?:reveals|shows|demonstrates)|"
    r"as proof of principle|we next"
    r")\b",
    re.IGNORECASE,
)
_METHODS_BODY_RE = re.compile(
    r"\b("
    r"we (?:used|performed|measured|calculated|applied|analyzed)|"
    r"samples were|protein databases|principal component analysis|"
    r"according to|experimental procedures|simulations? (?:were|used)"
    r")\b",
    re.IGNORECASE,
)
_DISCUSSION_BODY_RE = re.compile(
    r"\b("
    r"reviewed in|it is now clear|suggests that|we anticipate|in summary|"
    r"this review|together, these studies"
    r")\b",
    re.IGNORECASE,
)
_INLINE_SECTION_LABEL_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*)[.)]?\s+"
    r"(abstract|introduction|results?|discussion|methods?|materials and methods|conclusions?)\s*$",
    re.IGNORECASE,
)
_METADATA_BODY_RE = re.compile(
    r"\b(doi|corresponding author|e-mail address|available online|copyright|all rights reserved)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DecodedBlock:
    recovered_block: RecoveredBlock
    section: str
    subsection: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class SectionInference:
    section: str
    confidence: float
    rationale: str


@dataclass(frozen=True)
class SectionDecodeResult:
    recovered: StructureRecoveryResult
    blocks: list[DecodedBlock]

    def get(self, block_index: int) -> DecodedBlock:
        return self.blocks[block_index]

    def section_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for block in self.blocks:
            counts[block.section] = counts.get(block.section, 0) + 1
        return counts


@dataclass(frozen=True)
class _HeadingDecision:
    section: str
    subsection: str
    confidence: float
    rationale: str
    advance_state: bool


class BaseSectionDecoder:
    def decode(self, recovered: StructureRecoveryResult) -> SectionDecodeResult:
        raise NotImplementedError


class RuleBasedSectionDecoder(BaseSectionDecoder):
    def decode(self, recovered: StructureRecoveryResult) -> SectionDecodeResult:
        current_section = SECTION_OTHER
        current_subsection = ""
        abstract_seen = False

        decoded: list[DecodedBlock] = []
        for item in recovered.blocks:
            section = current_section
            subsection = current_subsection
            confidence = item.confidence
            rationale = "inherit previous section"

            if item.role == BlockRole.TITLE:
                section = SECTION_OTHER
                subsection = ""
                rationale = "title block"
                confidence = min(confidence, 0.95)

            elif item.role == BlockRole.FRONT_MATTER:
                section = SECTION_AUTHOR_INFO
                subsection = ""
                rationale = "front matter mapped to author info"

            elif item.role == BlockRole.ACK_META:
                section = SECTION_ACK
                subsection = ""
                rationale = "ack/meta region"
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.SUPPLEMENTARY_META:
                section = SECTION_SUPPLEMENTARY
                subsection = ""
                rationale = "supplementary region"
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.SUMMARY_BOX:
                section = SECTION_ABSTRACT if item.block.position_ratio <= 0.2 else SECTION_OTHER
                subsection = ""
                rationale = "summary box heading"
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.SECTION_HEADING:
                decoded_section = _decode_heading_section(item)
                section = decoded_section
                subsection = ""
                rationale = "major section heading"
                current_section = section
                current_subsection = subsection
                if section == SECTION_ABSTRACT:
                    abstract_seen = True

            elif item.role == BlockRole.SUBSECTION_HEADING:
                decoded_section = _decode_heading_section(item)
                if decoded_section == SECTION_OTHER:
                    decoded_section = _infer_section_from_context(
                        current_section=current_section,
                        position_ratio=item.block.position_ratio,
                        abstract_seen=abstract_seen,
                    )
                    rationale = "subsection inferred from context"
                    confidence = min(confidence, 0.72)
                else:
                    rationale = "subsection heading mapped directly"
                section = decoded_section
                subsection = item.block.text
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.REFERENCE_ENTRY:
                inline_section = _infer_from_inline_body_heading(item.block.text)
                if inline_section != SECTION_OTHER:
                    section = inline_section
                    subsection = ""
                    rationale = "inline section label recovered from reference-like paragraph"
                    current_section = section
                    current_subsection = subsection
                else:
                    section = SECTION_REFERENCES
                    subsection = ""
                    rationale = "reference entry"
                    current_section = section
                    current_subsection = subsection

            elif item.role in {BlockRole.FIGURE_CAPTION, BlockRole.TABLE_CAPTION}:
                section = current_section if current_section != SECTION_OTHER else SECTION_OTHER
                subsection = current_subsection
                rationale = "caption inherits local section"
                confidence = min(confidence, 0.8)

            elif item.role == BlockRole.BODY_PARAGRAPH:
                if (
                    current_section == SECTION_OTHER
                    and abstract_seen
                    and item.block.position_ratio <= 0.75
                ):
                    section = SECTION_RESULTS
                    rationale = "body in unresolved post-abstract region defaults to results"
                    confidence = min(confidence, 0.68)
                else:
                    section = current_section
                    rationale = "body inherits current section"
                subsection = current_subsection

            decoded.append(
                DecodedBlock(
                    recovered_block=item,
                    section=section,
                    subsection=subsection,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

        return SectionDecodeResult(recovered=recovered, blocks=decoded)


class LocalContextSectionDecoder(BaseSectionDecoder):
    def decode(self, recovered: StructureRecoveryResult) -> SectionDecodeResult:
        heading_counts = Counter(
            _normalize(item.block.text)
            for item in recovered.blocks
            if item.block.kind == BlockKind.HEADING
        )
        document_profile = _build_document_profile(recovered)

        current_section = SECTION_OTHER
        current_subsection = ""
        abstract_seen = False

        decoded: list[DecodedBlock] = []
        for index, item in enumerate(recovered.blocks):
            section = current_section
            subsection = current_subsection
            confidence = item.confidence
            rationale = "inherit previous section"

            if item.block.kind == BlockKind.HEADING:
                heading_decision = _decode_heading_with_context(
                    recovered=recovered,
                    index=index,
                    current_section=current_section,
                    abstract_seen=abstract_seen,
                    heading_counts=heading_counts,
                    document_profile=document_profile,
                )
                section = heading_decision.section
                subsection = heading_decision.subsection
                confidence = min(confidence, heading_decision.confidence)
                rationale = heading_decision.rationale

                if heading_decision.advance_state:
                    current_section = section
                    current_subsection = subsection
                    if section == SECTION_ABSTRACT:
                        abstract_seen = True

            elif item.role == BlockRole.FRONT_MATTER:
                section = SECTION_AUTHOR_INFO
                subsection = ""
                rationale = "front matter mapped to author info"

            elif item.role == BlockRole.ACK_META:
                section = SECTION_ACK
                subsection = ""
                rationale = "ack/meta region"
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.SUPPLEMENTARY_META:
                section = SECTION_SUPPLEMENTARY
                subsection = ""
                rationale = "supplementary region"
                current_section = section
                current_subsection = subsection

            elif item.role == BlockRole.REFERENCE_ENTRY:
                inline_section = _infer_from_inline_body_heading(item.block.text)
                if inline_section != SECTION_OTHER:
                    section = inline_section
                    subsection = ""
                    rationale = "inline section label recovered from reference-like paragraph"
                    current_section = section
                    current_subsection = subsection
                else:
                    section = SECTION_REFERENCES
                    subsection = ""
                    rationale = "reference entry"
                    current_section = section
                    current_subsection = subsection

            elif item.role in {BlockRole.FIGURE_CAPTION, BlockRole.TABLE_CAPTION}:
                section = current_section if current_section != SECTION_OTHER else SECTION_OTHER
                subsection = current_subsection
                rationale = "caption inherits local section"
                confidence = min(confidence, 0.8)

            elif item.role == BlockRole.BODY_PARAGRAPH:
                if current_section in {SECTION_OTHER, SECTION_AUTHOR_INFO}:
                    inferred = _infer_body_section_from_context(
                        recovered=recovered,
                        index=index,
                        abstract_seen=abstract_seen,
                        document_profile=document_profile,
                    )
                    if inferred.section != SECTION_OTHER and inferred.confidence >= 0.7:
                        section = inferred.section
                        rationale = "body inferred from local context"
                        confidence = min(confidence, inferred.confidence)
                        if inferred.rationale == "inline section label in body":
                            current_section = section
                            current_subsection = ""
                    else:
                        section = SECTION_OTHER
                        rationale = inferred.rationale
                        confidence = min(confidence, inferred.confidence)
                else:
                    section = current_section
                    rationale = "body inherits current section"
                subsection = current_subsection

            decoded.append(
                DecodedBlock(
                    recovered_block=item,
                    section=section,
                    subsection=subsection,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

        return SectionDecodeResult(recovered=recovered, blocks=decoded)


def decode_document_sections(
    recovered: StructureRecoveryResult,
    decoder: BaseSectionDecoder | None = None,
) -> SectionDecodeResult:
    effective_decoder = decoder or LocalContextSectionDecoder()
    return effective_decoder.decode(recovered)


def _decode_heading_section(item: RecoveredBlock) -> str:
    return _DIRECT_HEADING_SECTION_MAP.get(_normalize(item.block.text), SECTION_OTHER)


def _infer_section_from_context(
    *,
    current_section: str,
    position_ratio: float,
    abstract_seen: bool,
) -> str:
    if current_section in {
        SECTION_METHODS,
        SECTION_RESULTS,
        SECTION_DISCUSSION,
        SECTION_INTRO,
        SECTION_SUPPLEMENTARY,
        SECTION_ACK,
        SECTION_AUTHOR_INFO,
        SECTION_REFERENCES,
    }:
        return current_section

    if abstract_seen and 0.08 <= position_ratio <= 0.78:
        return SECTION_RESULTS

    if position_ratio < 0.2:
        return SECTION_INTRO

    if position_ratio > 0.85:
        return SECTION_DISCUSSION

    return SECTION_OTHER


def _decode_heading_with_context(
    *,
    recovered: StructureRecoveryResult,
    index: int,
    current_section: str,
    abstract_seen: bool,
    heading_counts: Counter[str],
    document_profile: dict[str, bool],
) -> _HeadingDecision:
    item = recovered.blocks[index]
    block = item.block
    normalized = _normalize(block.text)

    if _looks_like_page_marker(normalized) or _is_running_header(
        recovered=recovered,
        index=index,
        heading_counts=heading_counts,
    ):
        return _HeadingDecision(
            section=current_section,
            subsection=current_section != SECTION_OTHER and block.text or "",
            confidence=0.35,
            rationale="running header / page marker",
            advance_state=False,
        )

    if item.role == BlockRole.FRONT_MATTER or (
        item.role != BlockRole.TITLE and _looks_like_author_heading(block.text)
    ):
        return _HeadingDecision(
            section=SECTION_AUTHOR_INFO,
            subsection="",
            confidence=0.9,
            rationale="author/front-matter heading",
            advance_state=False,
        )

    if item.role == BlockRole.ACK_META:
        return _HeadingDecision(
            section=SECTION_ACK,
            subsection="",
            confidence=0.95,
            rationale="ack/meta heading",
            advance_state=True,
        )

    if item.role == BlockRole.SUPPLEMENTARY_META:
        return _HeadingDecision(
            section=SECTION_SUPPLEMENTARY,
            subsection="",
            confidence=0.95,
            rationale="supplementary heading",
            advance_state=True,
        )

    if item.role == BlockRole.SUMMARY_BOX:
        section = SECTION_ABSTRACT if block.position_ratio <= 0.2 else SECTION_OTHER
        return _HeadingDecision(
            section=section,
            subsection="",
            confidence=0.9,
            rationale="summary box heading",
            advance_state=section != SECTION_OTHER,
        )

    direct_section = _decode_heading_section(item)
    if direct_section != SECTION_OTHER:
        is_major = item.role == BlockRole.SECTION_HEADING or _looks_like_major_free_heading(
            block.text
        )
        return _HeadingDecision(
            section=direct_section,
            subsection="" if is_major else block.text,
            confidence=0.95 if item.role != BlockRole.UNKNOWN else 0.84,
            rationale="heading mapped directly",
            advance_state=True,
        )

    if item.role == BlockRole.TITLE or _looks_like_title_heading(recovered, index):
        return _HeadingDecision(
            section=SECTION_OTHER,
            subsection="",
            confidence=0.92,
            rationale="title block",
            advance_state=False,
        )

    inferred = _infer_heading_section_from_context(
        recovered=recovered,
        index=index,
        current_section=current_section,
        abstract_seen=abstract_seen,
        document_profile=document_profile,
    )
    is_major = (
        item.role == BlockRole.SECTION_HEADING
        or _looks_like_major_free_heading(block.text)
        or (
            document_profile["review_like"]
            and not abstract_seen
            and block.position_ratio >= 0.15
            and not _looks_like_author_heading(block.text)
        )
    )
    return _HeadingDecision(
        section=inferred.section if inferred.confidence >= 0.72 else SECTION_OTHER,
        subsection=(
            ""
            if is_major or inferred.section == SECTION_OTHER or inferred.confidence < 0.72
            else block.text
        ),
        confidence=inferred.confidence,
        rationale=inferred.rationale,
        advance_state=(
            inferred.section != SECTION_OTHER
            and inferred.confidence >= 0.72
            and not _looks_like_title_heading(recovered, index)
        ),
    )


def _infer_heading_section_from_context(
    *,
    recovered: StructureRecoveryResult,
    index: int,
    current_section: str,
    abstract_seen: bool,
    document_profile: dict[str, bool],
) -> SectionInference:
    item = recovered.blocks[index]
    block = item.block
    normalized = _normalize(block.text)

    cue_section = _infer_from_heading_text(normalized)
    if cue_section != SECTION_OTHER:
        return SectionInference(
            section=cue_section,
            confidence=0.84,
            rationale="heading inferred from lexical cue",
        )

    if current_section in {
        SECTION_METHODS,
        SECTION_RESULTS,
        SECTION_DISCUSSION,
        SECTION_INTRO,
        SECTION_CONCLUSION,
    }:
        return SectionInference(
            section=current_section,
            confidence=0.82,
            rationale="heading inherits active section context",
        )

    next_direct_section = _find_next_direct_heading_section(recovered, index)
    if next_direct_section == SECTION_RESULTS and block.position_ratio <= 0.2:
        return SectionInference(
            section=SECTION_INTRO,
            confidence=0.74,
            rationale="heading inferred from early pre-results position",
        )
    if next_direct_section == SECTION_METHODS and block.position_ratio >= 0.55:
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.73,
            rationale="heading inferred from late pre-methods position",
        )

    body_inference = _infer_from_local_body(recovered, index)
    if body_inference.section != SECTION_OTHER:
        return body_inference

    if document_profile["review_like"]:
        if block.position_ratio <= 0.25:
            return SectionInference(
                section=SECTION_INTRO,
                confidence=0.61,
                rationale="heading weakly inferred from early review position",
            )
        if block.position_ratio >= 0.82:
            return SectionInference(
                section=SECTION_CONCLUSION if "conclusion" in normalized else SECTION_DISCUSSION,
                confidence=0.6,
                rationale="heading weakly inferred from late review position",
            )
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.56,
            rationale="heading abstained: review-style fallback too weak",
        )

    fallback = _infer_section_from_context(
        current_section=current_section,
        position_ratio=block.position_ratio,
        abstract_seen=abstract_seen,
    )
    if fallback != SECTION_OTHER:
        return SectionInference(
            section=fallback,
            confidence=0.66 if abstract_seen else 0.6,
            rationale="heading abstained: positional fallback only",
        )

    if block.position_ratio <= 0.18:
        return SectionInference(
            section=SECTION_INTRO,
            confidence=0.56,
            rationale="heading abstained: weak early-position fallback",
        )
    if block.position_ratio >= 0.8:
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.56,
            rationale="heading abstained: weak late-position fallback",
        )
    return SectionInference(
        section=SECTION_OTHER,
        confidence=0.42,
        rationale="unresolved heading",
    )


def _infer_body_section_from_context(
    *,
    recovered: StructureRecoveryResult,
    index: int,
    abstract_seen: bool,
    document_profile: dict[str, bool],
) -> SectionInference:
    item = recovered.blocks[index]
    block = item.block

    inline_section = _infer_from_inline_body_heading(block.text)
    if inline_section != SECTION_OTHER:
        return SectionInference(
            section=inline_section,
            confidence=0.9,
            rationale="inline section label in body",
        )

    local_body_inference = _infer_from_local_body(recovered, index)
    if local_body_inference.section != SECTION_OTHER:
        return local_body_inference

    previous_direct = _find_previous_direct_heading_section(recovered, index)
    if previous_direct != SECTION_OTHER:
        return SectionInference(
            section=previous_direct,
            confidence=0.78,
            rationale="body inferred from previous direct heading",
        )

    next_direct = _find_next_direct_heading_section(recovered, index)
    if next_direct == SECTION_RESULTS and block.position_ratio <= 0.18:
        return SectionInference(
            section=SECTION_INTRO,
            confidence=0.72,
            rationale="body inferred from early pre-results region",
        )

    if document_profile["review_like"]:
        if (
            document_profile["has_title"]
            and 0.08 <= block.position_ratio <= 0.2
            and _looks_like_substantive_body(block.text)
        ):
            return SectionInference(
                section=SECTION_INTRO,
                confidence=0.72,
                rationale="body inferred from title-led review lede",
            )
        if block.position_ratio <= 0.2:
            return SectionInference(
                section=SECTION_INTRO,
                confidence=0.6,
                rationale="body abstained: weak early review fallback",
            )
        if block.position_ratio >= 0.82:
            return SectionInference(
                section=SECTION_CONCLUSION
                if "in summary" in block.text.lower()
                else SECTION_DISCUSSION,
                confidence=0.62 if "in summary" in block.text.lower() else 0.58,
                rationale="body abstained: weak late review fallback",
            )
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.54,
            rationale="body abstained: review-style fallback too weak",
        )

    fallback = _infer_section_from_context(
        current_section=SECTION_OTHER,
        position_ratio=block.position_ratio,
        abstract_seen=abstract_seen,
    )
    if fallback != SECTION_OTHER:
        return SectionInference(
            section=fallback,
            confidence=0.66 if abstract_seen else 0.58,
            rationale="body abstained: positional fallback only",
        )

    if block.position_ratio <= 0.18:
        return SectionInference(
            section=SECTION_INTRO,
            confidence=0.54,
            rationale="body abstained: weak early-position fallback",
        )
    if block.position_ratio >= 0.8:
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.54,
            rationale="body abstained: weak late-position fallback",
        )
    return SectionInference(
        section=SECTION_OTHER,
        confidence=0.42,
        rationale="body inherits unresolved section",
    )


def _infer_from_heading_text(normalized: str) -> str:
    if any(cue in normalized for cue in _RESULTS_HEADING_CUES):
        return SECTION_RESULTS
    if any(cue in normalized for cue in _METHODS_HEADING_CUES):
        return SECTION_METHODS
    if "conclusion" in normalized:
        return SECTION_CONCLUSION
    if any(cue in normalized for cue in _DISCUSSION_HEADING_CUES):
        return SECTION_DISCUSSION
    if any(cue in normalized for cue in _INTRO_HEADING_CUES):
        return SECTION_INTRO
    return SECTION_OTHER


def _infer_from_inline_body_heading(text: str) -> str:
    match = _INLINE_SECTION_LABEL_RE.match(" ".join(text.strip().split()))
    if not match:
        return SECTION_OTHER

    label = match.group(1).lower()
    if label == "abstract":
        return SECTION_ABSTRACT
    if label == "introduction":
        return SECTION_INTRO
    if label.startswith("result"):
        return SECTION_RESULTS
    if label == "discussion":
        return SECTION_DISCUSSION
    if label in {"method", "methods", "materials and methods"}:
        return SECTION_METHODS
    if label.startswith("conclusion"):
        return SECTION_CONCLUSION
    return SECTION_OTHER


def _infer_from_local_body(recovered: StructureRecoveryResult, index: int) -> SectionInference:
    snippets: list[str] = []
    for item in recovered.blocks[index + 1 :]:
        if item.block.kind == BlockKind.HEADING:
            break
        if item.role == BlockRole.BODY_PARAGRAPH:
            snippets.append(item.block.text)
        if len(snippets) >= 2:
            break

    if not snippets:
        return SectionInference(
            section=SECTION_OTHER,
            confidence=0.0,
            rationale="no local body evidence",
        )

    sample = " ".join(snippets)
    if _METHODS_BODY_RE.search(sample):
        return SectionInference(
            section=SECTION_METHODS,
            confidence=0.79,
            rationale="section inferred from local body cue",
        )
    if _RESULTS_BODY_RE.search(sample):
        return SectionInference(
            section=SECTION_RESULTS,
            confidence=0.79,
            rationale="section inferred from local body cue",
        )
    if _DISCUSSION_BODY_RE.search(sample):
        return SectionInference(
            section=SECTION_DISCUSSION,
            confidence=0.77,
            rationale="section inferred from local body cue",
        )
    return SectionInference(
        section=SECTION_OTHER,
        confidence=0.0,
        rationale="no local body evidence",
    )


def _find_next_direct_heading_section(recovered: StructureRecoveryResult, index: int) -> str:
    for item in recovered.blocks[index + 1 :]:
        if item.block.kind != BlockKind.HEADING:
            continue
        section = _decode_heading_section(item)
        if section != SECTION_OTHER:
            return section
    return SECTION_OTHER


def _find_previous_direct_heading_section(recovered: StructureRecoveryResult, index: int) -> str:
    for item in reversed(recovered.blocks[:index]):
        if item.block.kind != BlockKind.HEADING:
            continue
        section = _decode_heading_section(item)
        if section != SECTION_OTHER:
            return section
    return SECTION_OTHER


def _looks_like_title_heading(recovered: StructureRecoveryResult, index: int) -> bool:
    block = recovered.blocks[index].block
    normalized = _normalize(block.text)
    if index > 4:
        return False
    if index > 2 or block.position_ratio > 0.25:
        return False
    if any(previous.block.kind == BlockKind.HEADING for previous in recovered.blocks[:index]):
        return False
    if any(
        previous.block.kind != BlockKind.HEADING
        and previous.role == BlockRole.BODY_PARAGRAPH
        and previous.block.char_length >= 30
        for previous in recovered.blocks[:index]
    ):
        return False
    if len(normalized) < 12 or _looks_like_author_heading(block.text):
        return False
    if _looks_like_page_marker(normalized) or _RUNNING_HEADER_RE.search(block.text):
        return False
    return not ("copyright" in normalized or "doi" in normalized or "received" in normalized)


def _looks_like_author_heading(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if "@" in normalized:
        return True
    name_matches = _AUTHOR_NAME_RE.findall(normalized)
    has_author_joiners = (
        "," in normalized
        or " and " in normalized.lower()
        or "*" in normalized
        or bool(re.search(r"\d", normalized))
    )
    return len(name_matches) >= 2 and has_author_joiners


def _looks_like_substantive_body(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if len(normalized) < 80:
        return False
    if _METADATA_BODY_RE.search(normalized):
        return False
    if "@" in normalized or "www." in normalized.lower():
        return False
    alpha_count = sum(char.isalpha() for char in normalized)
    return alpha_count >= 50


def _looks_like_major_free_heading(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    words = normalized.split()
    if not (2 <= len(words) <= 12):
        return False
    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return uppercase_ratio >= 0.6


def _looks_like_page_marker(normalized: str) -> bool:
    return bool(_PAGE_MARKER_RE.fullmatch(normalized))


def _is_running_header(
    *,
    recovered: StructureRecoveryResult,
    index: int,
    heading_counts: Counter[str],
) -> bool:
    block = recovered.blocks[index].block
    normalized = _normalize(block.text)
    if _RUNNING_HEADER_RE.search(block.text):
        return True
    if heading_counts.get(normalized, 0) >= 3 and index > 3:
        return True
    return bool(_looks_like_author_heading(block.text) and block.position_ratio >= 0.2)


def _build_document_profile(recovered: StructureRecoveryResult) -> dict[str, bool]:
    direct_sections = {
        _decode_heading_section(item)
        for item in recovered.blocks
        if item.block.kind == BlockKind.HEADING
    }
    direct_sections.discard(SECTION_OTHER)

    return {
        "has_results": SECTION_RESULTS in direct_sections,
        "has_methods": SECTION_METHODS in direct_sections,
        "has_discussion": SECTION_DISCUSSION in direct_sections,
        "has_abstract": SECTION_ABSTRACT in direct_sections,
        "has_title": any(item.role == BlockRole.TITLE for item in recovered.blocks),
        "review_like": SECTION_RESULTS not in direct_sections
        and SECTION_METHODS not in direct_sections,
    }


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())
