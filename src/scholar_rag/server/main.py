from __future__ import annotations

import importlib.metadata
from typing import Any

from scholar_rag.server import tools

_MCP_MAJOR = int(importlib.metadata.version("mcp").split(".")[0])

if _MCP_MAJOR >= 2:
    from mcp.server.mcpserver import MCPServer as ServerV2

    ServerClass: type[Any] = ServerV2
else:
    from mcp.server.fastmcp import FastMCP as ServerV1  # type: ignore[attr-defined]

    ServerClass = ServerV1

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_chunks": "Semantic search over chunk bodies of a kb with metadata filters, "
    "embedding and rerank scores.",
    "search_documents": "Document-level PubMed-style search with optional FTS query and "
    "metadata filters.",
    "list_documents": "Paginated browse of documents in a kb.",
    "get_document": "Overview of a document: metadata, abstract, section outline and "
    "total character count.",
    "get_document_text": "Paginated reading of the full text or a single section of a document.",
    "add_document": "Asynchronously ingest a single PDF into an existing kb; returns a job_id.",
    "remove_document": "Synchronously delete a document from qdrant, catalog and disk.",
    "create_kb": "Asynchronously create a kb from all PDFs in a folder; returns a job_id.",
    "delete_kb": "Two-phase kb deletion: preview and confirm token, else full deletion.",
    "list_kbs": "List all knowledge bases with metadata and status.",
    "get_job": "Query the status, progress and result of a background job.",
}


def build_server() -> Any:
    server = ServerClass(name="scholar-rag-mcp")
    for tool_name in tools.TOOL_NAMES:
        server.add_tool(
            getattr(tools, f"tool_{tool_name}"),
            name=tool_name,
            description=_TOOL_DESCRIPTIONS[tool_name],
        )
    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
