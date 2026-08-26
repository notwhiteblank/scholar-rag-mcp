from __future__ import annotations

import re
from dataclasses import dataclass

_CAPTION_TEXT_RE = re.compile(
    r"^(?:\([A-Z]\)|fig(?:ure)?\.?\s*[A-Za-z]?\d+|table\s*[A-Za-z]?\d+)",
    re.IGNORECASE,
)
_CAPTION_CUE_RE = re.compile(
    r"\b(shown from top to bottom|left to right|scale bar|image z-stacks|cells were treated)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedParagraph:
    text: str
    section: str
    subsection: str
    content_type: str


@dataclass(frozen=True)
class AnnotationIssue:
    issue_type: str
    paragraph_index: int | None
    detail: str


@dataclass(frozen=True)
class RepairResult:
    paragraphs: tuple[ParsedParagraph, ...]
    issues: tuple[AnnotationIssue, ...]
    low_confidence: bool


def repair_paragraphs(
    paragraphs: list[ParsedParagraph],
    *,
    low_confidence_indices: list[int] | None = None,
) -> RepairResult:
    repaired: list[ParsedParagraph] = []
    issues: list[AnnotationIssue] = []
    low_confidence_set = set(low_confidence_indices or [])

    body_count = 0
    body_other_count = 0

    for index, paragraph in enumerate(paragraphs):
        updated = ParsedParagraph(
            text=paragraph.text,
            section=paragraph.section,
            subsection=paragraph.subsection,
            content_type=paragraph.content_type,
        )

        if updated.content_type == "body":
            body_count += 1
            if updated.section == "Other":
                body_other_count += 1

            if _looks_like_caption_noise(updated.text):
                updated = ParsedParagraph(
                    text=updated.text,
                    section=updated.section,
                    subsection=updated.subsection,
                    content_type="noise",
                )
                issues.append(
                    AnnotationIssue(
                        issue_type="caption_body_demoted",
                        paragraph_index=index,
                        detail=updated.text[:160],
                    )
                )

        repaired.append(updated)

    if body_count > 0 and body_other_count / body_count >= 0.2:
        issues.append(
            AnnotationIssue(
                issue_type="body_other_ratio_high",
                paragraph_index=None,
                detail=f"{body_other_count}/{body_count}",
            )
        )

    low_confidence = bool(low_confidence_set) or any(
        issue.issue_type in {"body_other_ratio_high", "caption_body_demoted"} for issue in issues
    )

    return RepairResult(
        paragraphs=tuple(repaired),
        issues=tuple(issues),
        low_confidence=low_confidence,
    )


def _looks_like_caption_noise(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return bool(_CAPTION_TEXT_RE.match(normalized) or _CAPTION_CUE_RE.search(normalized))
