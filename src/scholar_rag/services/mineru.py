from __future__ import annotations

import asyncio
import importlib
import inspect
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from glob import glob
from pathlib import Path
from typing import Any, cast

import httpx

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import PipelineStageError

_PARSER_MODULE = "mineru.cli.common"
_PARSER_ENTRY = "aio_do_parse"
_PIPELINE_BACKEND = "pipeline"
_CLI_TIMEOUT = 1800
_API_TIMEOUT = 1800
_MARKDOWN_KEYS = ("md_content", "markdown", "md")


def parse(pdf_path: Path) -> str:
    settings = Settings.load()
    backend = settings.mineru_backend
    try:
        if backend == "python":
            return _parse_via_python(pdf_path)
        if backend == "cli":
            return _parse_via_cli(pdf_path, settings.mineru_bin)
        return _parse_via_api(pdf_path, settings.mineru_api_url)
    except Exception as exc:
        raise PipelineStageError(f"mineru backend={backend} failed: {exc}") from exc


def _parse_via_python(pdf_path: Path) -> str:
    common = importlib.import_module(_PARSER_MODULE)
    entry = getattr(common, _PARSER_ENTRY)
    output_dir = Path(tempfile.mkdtemp(prefix="mineru_out_"))
    kwargs = _build_entry_kwargs(cast(Callable[..., Any], entry), pdf_path, output_dir)
    result: Any = entry(**kwargs)
    if inspect.isawaitable(result):
        result = asyncio.run(_coerce_awaitable(result))
    return _read_parse_output(output_dir, pdf_path.stem)


async def _coerce_awaitable(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _build_entry_kwargs(entry: Callable[..., Any], pdf_path: Path, output_dir: Path) -> dict[str, Any]:
    parameters = inspect.signature(entry).parameters
    if "pdf_file_names" in parameters:
        return {
            "output_dir": str(output_dir),
            "pdf_file_names": [pdf_path.name],
            "pdf_bytes_list": [pdf_path.read_bytes()],
            "p_lang_list": ["en"],
            "backend": _PIPELINE_BACKEND,
            "formula_enable": True,
            "table_enable": True,
        }
    base: dict[str, object] = {
        "pdf_file": str(pdf_path),
        "output_dir": str(output_dir),
        "backend": _PIPELINE_BACKEND,
        "lang_list": ["en"],
        "formula_enable": True,
        "table_enable": True,
    }
    return {key: value for key, value in base.items() if key in parameters}


def _parse_via_cli(pdf_path: Path, mineru_bin: str) -> str:
    output_dir = Path(tempfile.mkdtemp(prefix="mineru_cli_"))
    completed = subprocess.run(
        [mineru_bin, "-p", str(pdf_path), "-o", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cli exit {completed.returncode}: {detail}")
    return _read_parse_output(output_dir, pdf_path.stem)


def _parse_via_api(pdf_path: Path, api_url: str) -> str:
    try:
        with pdf_path.open("rb") as handle:
            resp = httpx.post(
                f"{api_url}/file_parse",
                files={"files": (pdf_path.name, handle, "application/pdf")},
                data={
                    "backend": _PIPELINE_BACKEND,
                    "lang_list": "en",
                    "parse_method": "auto",
                    "return_md": "true",
                    "formula_enable": "true",
                    "table_enable": "true",
                },
                timeout=_API_TIMEOUT,
            )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"api request failed: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"api status {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError("api returned non-json body") from exc
    markdown = _extract_markdown(payload, pdf_path.stem)
    if markdown is None:
        raise RuntimeError("no md content in api response")
    return markdown


def _extract_markdown(data: object, stem: str) -> str | None:
    if not isinstance(data, dict):
        return None
    file_results = data.get("results", {})
    if not isinstance(file_results, dict):
        return None
    file_data = file_results.get(stem)
    if not isinstance(file_data, dict) and len(file_results) == 1:
        only_value = next(iter(file_results.values()))
        if isinstance(only_value, dict):
            file_data = only_value
    if not isinstance(file_data, dict):
        return None
    for key in _MARKDOWN_KEYS:
        content = file_data.get(key)
        if isinstance(content, str) and content:
            return content
    return None


def _read_parse_output(output_dir: Path, stem: str) -> str:
    paths: list[Path] = [output_dir / stem / "auto" / f"{stem}.md"]
    for item in sorted(glob(str(output_dir / "**" / "*.md"), recursive=True)):
        paths.append(Path(item))
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if resolved.is_file():
            text = resolved.read_text(encoding="utf-8")
            if text.strip():
                return text
    raise RuntimeError("parse produced no markdown output")
