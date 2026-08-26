from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scholar_rag.core.errors import ServiceUnavailableError  # noqa: E402
from scholar_rag.ingest.annotate import stage_annotate  # noqa: E402
from scholar_rag.ingest.annotate.document_structure import extract_document_blocks  # noqa: E402
from scholar_rag.ingest.annotate.resolver import SectionResolver  # noqa: E402
from scholar_rag.ingest.annotate.section_decode import (  # noqa: E402
    SECTION_OTHER,
    decode_document_sections,
)
from scholar_rag.ingest.annotate.structure_recovery import (  # noqa: E402
    recover_document_structure,
)
from tests.fixtures.gen_pdfs import PAPER_A_BLOCKS, to_markdown  # noqa: E402


def _source_to_decoded(markdown: str):
    structured = extract_document_blocks(markdown)
    recovered = recover_document_structure(structured)
    return decode_document_sections(recovered)

PAPER_A_MD = to_markdown(PAPER_A_BLOCKS)


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_BASE_URL", "http://chat.test/v1")


class FakeChatClient:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def complete(self, prompt: str, *, json_schema: dict, max_retries: int = 2):
        self.calls.append((prompt, json_schema))
        if not self._responses:
            raise ServiceUnavailableError("no more responses")
        return self._responses.pop(0)

    def __call__(self, *args: object, **kwargs: object):
        return self


def _overrides_response(block_index: int, section: str) -> dict:
    return {
        "items": [
            {
                "block_index": block_index,
                "decision": "override",
                "section_type": section,
                "confidence": 0.9,
                "clear_subsection": False,
                "rationale": "looks like a results finding",
            }
        ]
    }


def test_missing_candidates_returns_unchanged() -> None:
    source = "# Title\n\nA deterministic label paragraph\n\nShort."
    recovered = _source_to_decoded(source)
    resolver = SectionResolver()
    result = resolver.resolve(recovered, chat_client=FakeChatClient([]), cache_dir=None)
    assert result.blocks == recovered.blocks


def test_resolver_override_changes_section() -> None:
    source = "# Title\n\nBody paragraph that stays in other.\n\nBackground on the subject."
    recovered = _source_to_decoded(source)
    candidates = [block for block in recovered.blocks if block.section == SECTION_OTHER]
    assert candidates, "expected at least one low-confidence Other block"
    target = candidates[0].recovered_block.block.index
    fake = FakeChatClient([_overrides_response(target, "Discussion")])
    resolver = SectionResolver()
    result = resolver.resolve(recovered, chat_client=fake, cache_dir=None)
    by_index = {b.recovered_block.block.index: b.section for b in result.blocks}
    assert by_index[target] == "Discussion"
    assert fake.calls and "keep_current" in fake.calls[0][0]


def test_resolver_abstain_keeps_original() -> None:
    source = "# Title\n\nBody paragraph that stays in other.\n\nBackground on the subject."
    recovered = _source_to_decoded(source)
    candidates = [block for block in recovered.blocks if block.section == SECTION_OTHER]
    target = candidates[0].recovered_block.block.index
    original = recovered.get(target).section
    fake = FakeChatClient(
        [
            {
                "items": [
                    {
                        "block_index": target,
                        "decision": "abstain",
                        "section_type": "Other",
                        "confidence": 0.5,
                        "clear_subsection": False,
                        "rationale": "insufficient evidence",
                    }
                ]
            }
        ]
    )
    resolver = SectionResolver()
    result = resolver.resolve(recovered, chat_client=fake, cache_dir=None)
    assert result.get(target).section == original


def test_resolver_caches_blake2b_results(tmp_path: Path) -> None:
    source = "# Title\n\nBody paragraph that stays in other.\n\nBackground on the subject."
    recovered = _source_to_decoded(source)
    candidates = [block for block in recovered.blocks if block.section == SECTION_OTHER]
    target = candidates[0].recovered_block.block.index
    fake = FakeChatClient([_overrides_response(target, "Discussion")])
    cache_dir = tmp_path / "resolver_cache"
    resolver = SectionResolver()
    resolver.resolve(recovered, chat_client=fake, cache_dir=cache_dir)
    assert fake.calls
    resolver.resolve(recovered, chat_client=fake, cache_dir=cache_dir)
    assert len(fake.calls) == 1, "second call should hit the on-disk cache"
    cache_files = list(cache_dir.rglob("*.json"))
    assert len(cache_files) == 1


def test_resolver_client_failure_is_silent() -> None:
    source = "# Title\n\nBody paragraph that stays in other.\n\nBackground on the subject."
    recovered = _source_to_decoded(source)

    class BrokenChat:
        def complete(self, prompt: str, *, json_schema: dict, max_retries: int = 2):
            raise ServiceUnavailableError("chat down")

    resolver = SectionResolver()
    result = resolver.resolve(recovered, chat_client=BrokenChat(), cache_dir=None)
    assert result.blocks == recovered.blocks


def test_stage_annotate_runs_resolver_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "true")
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path))
    fake = FakeChatClient([])

    def fake_get_chat_client() -> FakeChatClient:
        return fake

    monkeypatch.setattr("scholar_rag.ingest.annotate.get_chat_client", fake_get_chat_client)
    doc = stage_annotate(PAPER_A_MD)
    assert doc.blocks is not None


def test_stage_annotate_skips_resolver_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED", "false")

    def boom() -> None:
        raise AssertionError("resolver should not run")

    monkeypatch.setattr("scholar_rag.ingest.annotate.get_chat_client", boom)
    doc = stage_annotate(PAPER_A_MD)
    assert doc.blocks is not None
