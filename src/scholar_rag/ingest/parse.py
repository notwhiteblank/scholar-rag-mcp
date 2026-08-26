from __future__ import annotations

from pathlib import Path

import scholar_rag.services.mineru as mineru
from scholar_rag.ingest.staging import StagedPdf


def stage_parse(staged: StagedPdf, cache_dir: Path) -> str:
    cache_dir = Path(cache_dir)
    md_path = cache_dir / f"{staged.content_hash}.md"
    if md_path.is_file():
        return md_path.read_text(encoding="utf-8")
    markdown = mineru.parse(staged.path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    return markdown
