from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scholar_rag.ingest.annotate import stage_annotate  # noqa: E402
from tests.fixtures.gen_pdfs import (  # noqa: E402
    PAPER_A_BLOCKS,
    PAPER_B_BLOCKS,
    PAPER_C_BLOCKS,
    to_markdown,
)

GOLDEN_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "golden"

PAPER_A_MD = to_markdown(PAPER_A_BLOCKS)
PAPER_B_MD = to_markdown(PAPER_B_BLOCKS)
PAPER_C_MD = to_markdown(PAPER_C_BLOCKS)


@pytest.fixture(autouse=True)
def _deterministic_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "false")


def test_empty_text_yields_empty_document() -> None:
    doc = stage_annotate("")
    assert doc.blocks == ()
    assert doc.sections == ()


def test_heading_recognition_preserves_levels() -> None:
    doc = stage_annotate(PAPER_A_MD)
    assert doc.blocks
    headings = [b for b in doc.blocks if b.is_heading]
    assert headings
    levels = {b.heading_level for b in headings}
    assert levels == {1, 2, 3}
    architecture = [b for b in headings if b.text == "Model Architecture"]
    assert architecture and architecture[0].heading_level == 3


def test_author_line_and_reference_entries_are_filtered() -> None:
    doc = stage_annotate(PAPER_A_MD)
    joined = "\n".join(b.text for b in doc.blocks)
    assert "First Author" not in joined
    assert "Smith J" not in joined
    assert "Rossi M" not in joined


def test_canonical_sections_are_identified() -> None:
    doc = stage_annotate(PAPER_A_MD)
    sections = {name for name, count in doc.sections}
    assert {"Abstract", "Introduction", "Methods", "Results", "Discussion", "References"} <= sections


def test_section_character_counts_are_positive_for_body_sections() -> None:
    doc = stage_annotate(PAPER_A_MD)
    counts = dict(doc.sections)
    assert counts["Methods"] >= 100
    assert counts["Results"] >= 100
    assert counts["References"] == 0


def test_review_paper_summary_maps_to_abstract() -> None:
    doc = stage_annotate(PAPER_B_MD)
    sections = dict(doc.sections)
    assert sections.get("Abstract", 0) >= 100
    assert "References" in sections


def test_numbered_headings_are_handled() -> None:
    doc = stage_annotate(PAPER_C_MD)
    sections = dict(doc.sections)
    assert sections.get("Introduction", 0) >= 50
    assert sections.get("Abstract", 0) >= 50
    assert any(b.is_heading and b.text == "3. Experiments" for b in doc.blocks)


def test_deterministic_across_runs() -> None:
    first = stage_annotate(PAPER_A_MD)
    second = stage_annotate(PAPER_A_MD)
    assert first == second


def test_blocks_are_frozen_and_typed() -> None:
    doc = stage_annotate(PAPER_A_MD)
    for block in doc.blocks:
        assert block.is_heading is False or block.heading_level >= 1
        assert block.confidence >= 0.0
        assert block.section


def test_sections_sum_matches_body_character_count() -> None:
    doc = stage_annotate(PAPER_A_MD)
    body_total = sum(len(b.text) for b in doc.blocks if not b.is_heading)
    section_total = sum(count for _name, count in doc.sections)
    assert body_total == section_total


@pytest.mark.parametrize("name", ["paper_a", "paper_b", "paper_c"])
def test_golden_matches(name: str) -> None:
    markdown = {
        "paper_a": PAPER_A_MD,
        "paper_b": PAPER_B_MD,
        "paper_c": PAPER_C_MD,
    }[name]
    golden = json.loads((GOLDEN_DIR / f"annotate_{name}.json").read_text(encoding="utf-8"))
    doc = stage_annotate(markdown)
    actual = {
        "blocks": [
            {
                "text": block.text,
                "is_heading": block.is_heading,
                "heading_level": block.heading_level,
                "section": block.section,
                "confidence": block.confidence,
            }
            for block in doc.blocks
        ],
        "sections": [[name_, count] for name_, count in doc.sections],
    }
    assert actual == golden
