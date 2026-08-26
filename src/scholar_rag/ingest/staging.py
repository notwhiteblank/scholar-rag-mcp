from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from scholar_rag.core.errors import PipelineStageError

_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class StagedPdf:
    path: Path
    source_path: Path
    content_hash: str
    name: str


@dataclass(frozen=True)
class ChunkConfig:
    min_chunk: int = 300
    max_chunk: int = 1500
    overlap: int = 100


def stage_pdf(pdf_path: Path, work_dir: Path) -> StagedPdf:
    pdf_path = Path(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise PipelineStageError(f"not a pdf file: {pdf_path}")
    if not pdf_path.is_file():
        raise PipelineStageError(f"pdf not found: {pdf_path}")
    raw = pdf_path.read_bytes()
    if not raw.startswith(_PDF_MAGIC):
        raise PipelineStageError(f"missing %PDF- magic: {pdf_path}")
    content_hash = hashlib.sha256(raw).hexdigest()
    name = pdf_path.stem
    target = Path(work_dir) / f"{name}.pdf"
    if target != pdf_path:
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf_path, target)
    return StagedPdf(path=target, source_path=pdf_path, content_hash=content_hash, name=name)
