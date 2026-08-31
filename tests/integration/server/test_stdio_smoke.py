import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scholar_rag.core.registry import Registry

pytestmark = pytest.mark.integration

EXPECTED_TOOLS = [
    "search_chunks",
    "search_documents",
    "list_documents",
    "get_document",
    "get_document_text",
    "add_document",
    "remove_document",
    "create_kb",
    "delete_kb",
    "list_kbs",
    "get_job",
]

_TIMEOUT = 60.0


def _server_env(root: Path, qdrant_url: str, model_url: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SCHOLAR_RAG_"):
            del env[key]
    env["HOME"] = str(root)
    env["USERPROFILE"] = str(root)
    env["XDG_DATA_HOME"] = str(root / "xdg-data")
    env["LOCALAPPDATA"] = str(root / "AppData" / "Local")
    env.pop("APPDATA", None)
    env["SCHOLAR_RAG_DATA_DIR"] = str(root / "data")
    env["SCHOLAR_RAG_QDRANT_URL"] = qdrant_url
    env["SCHOLAR_RAG_EMBED_BACKEND"] = "api"
    env["SCHOLAR_RAG_EMBED_BASE_URL"] = model_url
    env["SCHOLAR_RAG_RERANK_BACKEND"] = "api"
    env["SCHOLAR_RAG_RERANK_BASE_URL"] = model_url
    env["SCHOLAR_RAG_ANNOTATION_RESOLVER_ENABLED"] = "false"
    env["SCHOLAR_RAG_KEYWORDS_ENABLED"] = "false"
    env["SCHOLAR_RAG_CROSSREF_ENABLED"] = "false"
    return env


@pytest.fixture
def seed_kb(monkeypatch, tmp_path) -> str:
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(tmp_path / "data"))
    Registry().create_kb(
        "smoke",
        {
            "name": "smoke",
            "created_at": "2024-05-01T00:00:00Z",
            "dim": 8,
            "embedding_model": "fake",
            "chunk": {"min": 300, "max": 1500, "overlap": 100},
            "schema_version": 1,
        },
    )
    return "smoke"


@pytest.mark.asyncio
async def test_stdio_server_initialize_list_tools_and_list_kbs(
    tmp_path, qdrant_instance, fake_model_url, seed_kb
):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "scholar_rag.server.main"],
        env=_server_env(tmp_path, qdrant_instance["url"], fake_model_url),
    )
    async with AsyncExitStack() as stack:
        streams = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(*streams))
        await asyncio.wait_for(session.initialize(), timeout=_TIMEOUT)
        tools = await asyncio.wait_for(session.list_tools(), timeout=_TIMEOUT)
        names = [tool.name for tool in tools.tools]
        assert names == EXPECTED_TOOLS
        result = await asyncio.wait_for(session.call_tool("list_kbs", {}), timeout=_TIMEOUT)
        assert result.is_error is False
        payload = result.structured_content
        assert payload == {
            "kbs": [
                {
                    "name": "smoke",
                    "doc_count": 0,
                    "chunk_count": 0,
                    "created_at": "2024-05-01T00:00:00Z",
                    "status": "ready",
                }
            ]
        }
