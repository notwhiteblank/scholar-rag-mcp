from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = FIXTURES_DIR / "golden"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scholar_rag.ingest.annotate import stage_annotate  # noqa: E402
from tests.fixtures.gen_pdfs import (  # noqa: E402
    PAPER_A_BLOCKS,
    PAPER_B_BLOCKS,
    PAPER_C_BLOCKS,
    to_markdown,
)

PAPERS = {
    "paper_a": PAPER_A_BLOCKS,
    "paper_b": PAPER_B_BLOCKS,
    "paper_c": PAPER_C_BLOCKS,
}


def generate_goldens(output_dir: Path | None = None) -> dict[str, Path]:
    os.environ["SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED"] = "false"
    target_dir = Path(output_dir) if output_dir is not None else GOLDEN_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, blocks in PAPERS.items():
        markdown = to_markdown(blocks)
        doc = stage_annotate(markdown)
        payload = {
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
        out_path = target_dir / f"annotate_{name}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written[name] = out_path
    return written


def main() -> None:
    written = generate_goldens()
    print(f"Generated {len(written)} golden files in {GOLDEN_DIR}:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    sys.exit(main())
