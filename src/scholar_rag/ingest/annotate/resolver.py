from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from scholar_rag.ingest.annotate.document_structure import BlockKind, BlockRole
from scholar_rag.ingest.annotate.section_decode import (
    SECTION_OTHER,
    DecodedBlock,
    SectionDecodeResult,
)


class ChatClientLike(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None,
        max_retries: int = 2,
    ) -> str | dict[str, Any]:
        ...

SECTION_RESOLUTION_PROMPT = """You are resolving low-confidence scientific paper section labels.

For each candidate block below, decide ONE of:
- keep_current: the current section is already acceptable
- override: replace the current section with a better coarse section label
- abstain: evidence is insufficient, do not change the current label

Available coarse section labels:
{section_options}

Rules:
- Prefer abstain over guessing.
- Only use override when local evidence is genuinely strong.
- Use keep_current when the current section is reasonable even if imperfect.
- Treat obvious metadata, DOI lines, author lines, and running headers as abstain.
- Results-like topical findings usually belong to Results.
- Review-style conceptual headings often belong to Discussion, but only if evidence is clear.
- Methods requires concrete experimental or computational procedure evidence.

Return JSON:
[
  {{
    "block_index": 12,
    "decision": "override",
    "section_type": "Results",
    "confidence": 0.86,
    "clear_subsection": false,
    "rationale": "finding-oriented heading plus result-like local body"
  }}
]

Candidates:
{candidate_text}
"""

SECTION_TYPES = [
    "Abstract",
    "Introduction",
    "Results",
    "Discussion",
    "Methods",
    "Conclusion",
    "References",
    "Supplementary",
    "Acknowledgments",
    "Author_Info",
    "Other",
]


@dataclass(frozen=True)
class SectionResolutionCandidate:
    block_index: int
    role: str
    current_section: str
    current_subsection: str
    current_confidence: float
    text: str
    previous_heading: str
    next_heading: str
    local_body_excerpt: str


class SectionResolution(BaseModel):
    block_index: int
    decision: Literal["keep_current", "override", "abstain"]
    section_type: Literal[
        "Abstract",
        "Introduction",
        "Results",
        "Discussion",
        "Methods",
        "Conclusion",
        "References",
        "Supplementary",
        "Acknowledgments",
        "Author_Info",
        "Other",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    clear_subsection: bool = False
    rationale: str


class SectionResolutionList(BaseModel):
    items: list[SectionResolution]


def resolution_schema() -> dict[str, Any]:
    return {
        "title": "SectionResolutionList",
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "block_index": {"type": "integer"},
                        "decision": {
                            "enum": ["keep_current", "override", "abstain"],
                            "type": "string",
                        },
                        "section_type": {"enum": list(SECTION_TYPES), "type": "string"},
                        "confidence": {"type": "number"},
                        "clear_subsection": {"type": "boolean"},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "block_index",
                        "decision",
                        "section_type",
                        "confidence",
                        "clear_subsection",
                        "rationale",
                    ],
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


class SectionResolver:
    def __init__(
        self,
        *,
        candidate_threshold: float = 0.7,
        apply_confidence: float = 0.8,
        max_candidates: int = 24,
        cache_version: str = "resolver-v1",
    ) -> None:
        self._candidate_threshold = candidate_threshold
        self._apply_confidence = apply_confidence
        self._max_candidates = max_candidates
        self._cache_version = cache_version

    def resolve(
        self,
        decoded: SectionDecodeResult,
        *,
        chat_client: ChatClientLike | None = None,
        cache_dir: Path | None = None,
    ) -> SectionDecodeResult:
        candidates = collect_low_confidence_candidates(
            decoded,
            threshold=self._candidate_threshold,
            max_candidates=self._max_candidates,
        )
        if not candidates or chat_client is None:
            return decoded
        resolutions = self._resolve_candidates(candidates, chat_client, cache_dir)
        if not resolutions:
            return decoded

        by_index = {item.block_index: item for item in resolutions}
        updated_blocks: list[DecodedBlock] = []
        for item in decoded.blocks:
            resolution = by_index.get(item.recovered_block.block.index)
            if resolution is None:
                updated_blocks.append(item)
                continue
            updated_blocks.append(_apply_resolution(item, resolution, self._apply_confidence))

        return SectionDecodeResult(recovered=decoded.recovered, blocks=updated_blocks)

    def _resolve_candidates(
        self,
        candidates: list[SectionResolutionCandidate],
        chat_client: ChatClientLike,
        cache_dir: Path | None,
    ) -> list[SectionResolution]:
        cache_key = _build_cache_key(
            candidates,
            cache_version=self._cache_version,
            threshold=self._candidate_threshold,
            apply_confidence=self._apply_confidence,
        )
        cached = _load_resolutions(cache_key, cache_dir, self._cache_version)
        if cached is not None:
            return cached

        prompt = SECTION_RESOLUTION_PROMPT.format(
            section_options="\n".join(f"- {item}" for item in SECTION_TYPES),
            candidate_text=_format_candidates(candidates),
        )
        try:
            result = chat_client.complete(prompt, json_schema=resolution_schema(), max_retries=2)
        except Exception:
            return []
        if not isinstance(result, dict):
            return []
        items = result.get("items")
        if not isinstance(items, list):
            return []
        resolutions: list[SectionResolution] = []
        for item in items:
            try:
                resolutions.append(SectionResolution.model_validate(item))
            except ValidationError:
                continue
        if resolutions:
            _save_resolutions(cache_key, cache_dir, self._cache_version, resolutions)
        return resolutions


def collect_low_confidence_candidates(
    decoded: SectionDecodeResult,
    *,
    threshold: float = 0.7,
    max_candidates: int = 24,
) -> list[SectionResolutionCandidate]:
    normalized_counts = Counter(
        _normalize_candidate_text(item.recovered_block.block.text)
        for item in decoded.blocks
        if _normalize_candidate_text(item.recovered_block.block.text)
    )
    lead_labels = {
        _normalize_candidate_text(item.recovered_block.block.text)
        for item in decoded.blocks[:5]
        if _normalize_candidate_text(item.recovered_block.block.text)
    }
    candidates: list[SectionResolutionCandidate] = []
    for index, item in enumerate(decoded.blocks):
        block = item.recovered_block.block
        role = item.recovered_block.role
        if len(candidates) >= max_candidates:
            break

        if not _is_candidate_block(
            item,
            threshold,
            normalized_counts=normalized_counts,
            lead_labels=lead_labels,
        ):
            continue

        candidates.append(
            SectionResolutionCandidate(
                block_index=block.index,
                role=role.value,
                current_section=item.section,
                current_subsection=item.subsection,
                current_confidence=item.confidence,
                text=block.text,
                previous_heading=_find_previous_heading(decoded, index),
                next_heading=_find_next_heading(decoded, index),
                local_body_excerpt=_collect_local_body_excerpt(decoded, index),
            )
        )

    return candidates


def _is_candidate_block(
    item: DecodedBlock,
    threshold: float,
    *,
    normalized_counts: Counter[str],
    lead_labels: set[str],
) -> bool:
    del threshold
    role = item.recovered_block.role
    block = item.recovered_block.block
    if item.rationale == "title block":
        return False

    if role in {
        BlockRole.TITLE,
        BlockRole.ACK_META,
        BlockRole.SUPPLEMENTARY_META,
        BlockRole.FRONT_MATTER,
        BlockRole.REFERENCE_ENTRY,
        BlockRole.FIGURE_CAPTION,
        BlockRole.TABLE_CAPTION,
    }:
        return False

    if _looks_like_running_header_candidate(
        item,
        normalized_counts=normalized_counts,
        lead_labels=lead_labels,
    ):
        return False

    if item.section != SECTION_OTHER:
        return False

    if block.kind == BlockKind.HEADING:
        return True

    return role == BlockRole.BODY_PARAGRAPH


def _normalize_candidate_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _looks_like_running_header_candidate(
    item: DecodedBlock,
    *,
    normalized_counts: Counter[str],
    lead_labels: set[str],
) -> bool:
    block = item.recovered_block.block
    normalized = _normalize_candidate_text(block.text)
    if not normalized:
        return False
    if re.fullmatch(r"\d+\s+of\s+\d+", normalized):
        return True
    if block.position_ratio >= 0.2 and _looks_like_author_like_line(block.text):
        return True
    if block.index >= 3 and len(normalized) <= 80 and normalized in lead_labels:
        return True
    return bool(block.index > 3 and len(normalized) <= 60 and normalized_counts[normalized] >= 2)


def _looks_like_author_like_line(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    if "@" in normalized:
        return True
    name_matches = re.findall(
        r"[A-Z][A-Za-z'`-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'`-]+)+(?:\d+|\*)?",
        normalized,
    )
    has_author_joiners = (
        "," in normalized
        or " and " in normalized.lower()
        or "*" in normalized
        or bool(re.search(r"\d", normalized))
    )
    return len(name_matches) >= 2 and has_author_joiners


def _find_previous_heading(decoded: SectionDecodeResult, index: int) -> str:
    for item in reversed(decoded.blocks[:index]):
        if item.recovered_block.block.kind == BlockKind.HEADING:
            return item.recovered_block.block.text
    return ""


def _find_next_heading(decoded: SectionDecodeResult, index: int) -> str:
    for item in decoded.blocks[index + 1 :]:
        if item.recovered_block.block.kind == BlockKind.HEADING:
            return item.recovered_block.block.text
    return ""


def _collect_local_body_excerpt(decoded: SectionDecodeResult, index: int) -> str:
    snippets: list[str] = []
    for item in decoded.blocks[index + 1 :]:
        if item.recovered_block.block.kind == BlockKind.HEADING:
            break
        if item.recovered_block.role == BlockRole.BODY_PARAGRAPH:
            snippets.append(item.recovered_block.block.text[:220])
        if len(snippets) >= 2:
            break
    return "\n".join(snippets)


def _format_candidates(candidates: list[SectionResolutionCandidate]) -> str:
    parts: list[str] = []
    for item in candidates:
        parts.append(
            "\n".join(
                [
                    f"[block_index={item.block_index}]",
                    f"role: {item.role}",
                    f"current_section: {item.current_section}",
                    f"current_subsection: {item.current_subsection or '(none)'}",
                    f"current_confidence: {item.current_confidence:.2f}",
                    f"text: {item.text}",
                    f"previous_heading: {item.previous_heading or '(none)'}",
                    f"next_heading: {item.next_heading or '(none)'}",
                    f"local_body_excerpt: {item.local_body_excerpt or '(none)'}",
                ]
            )
        )
    return "\n\n".join(parts)


def _apply_resolution(
    item: DecodedBlock,
    resolution: SectionResolution,
    apply_confidence: float,
) -> DecodedBlock:
    if resolution.decision == "abstain":
        return item

    if resolution.decision == "keep_current":
        if resolution.confidence < apply_confidence:
            return item
        return DecodedBlock(
            recovered_block=item.recovered_block,
            section=item.section,
            subsection=item.subsection,
            confidence=max(item.confidence, resolution.confidence),
            rationale=f"llm keep_current: {resolution.rationale}",
        )

    if resolution.confidence < apply_confidence:
        return item

    subsection = "" if resolution.clear_subsection else item.subsection
    return DecodedBlock(
        recovered_block=item.recovered_block,
        section=resolution.section_type,
        subsection=subsection,
        confidence=resolution.confidence,
        rationale=f"llm override: {resolution.rationale}",
    )


def _build_cache_key(
    candidates: list[SectionResolutionCandidate],
    *,
    cache_version: str,
    threshold: float,
    apply_confidence: float,
) -> str:
    payload = {
        "cache_version": cache_version,
        "candidate_threshold": threshold,
        "apply_confidence": apply_confidence,
        "candidates": [
            {
                "block_index": item.block_index,
                "role": item.role,
                "current_section": item.current_section,
                "current_subsection": item.current_subsection,
                "current_confidence": round(item.current_confidence, 4),
                "text": item.text,
                "previous_heading": item.previous_heading,
                "next_heading": item.next_heading,
                "local_body_excerpt": item.local_body_excerpt,
            }
            for item in candidates
        ],
    }
    return hashlib.blake2b(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def _cache_path(cache_key: str, cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    prefix = cache_key[:2]
    nested = Path(cache_dir) / prefix
    nested.mkdir(parents=True, exist_ok=True)
    return nested / f"{cache_key}.json"


def _load_resolutions(
    cache_key: str,
    cache_dir: Path | None,
    cache_version: str,
) -> list[SectionResolution] | None:
    path = _cache_path(cache_key, cache_dir)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("cache_version") != cache_version:
        return None
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    resolutions: list[SectionResolution] = []
    for item in items:
        try:
            resolutions.append(SectionResolution.model_validate(item))
        except ValidationError:
            continue
    return resolutions


def _save_resolutions(
    cache_key: str,
    cache_dir: Path | None,
    cache_version: str,
    resolutions: list[SectionResolution],
) -> None:
    path = _cache_path(cache_key, cache_dir)
    if path is None:
        return
    payload = {
        "cache_version": cache_version,
        "items": [item.model_dump(mode="json") for item in resolutions],
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        return
