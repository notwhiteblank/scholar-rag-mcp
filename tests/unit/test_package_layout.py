import importlib
import pkgutil

import pytest

import scholar_rag

STUB_MODULES = [
    "scholar_rag.core",
    "scholar_rag.core.config",
    "scholar_rag.core.errors",
    "scholar_rag.core.jobs",
    "scholar_rag.core.log",
    "scholar_rag.core.registry",
    "scholar_rag.core.types",
    "scholar_rag.models.base",
    "scholar_rag.models.chat",
    "scholar_rag.models.embedding",
    "scholar_rag.models.rerank",
    "scholar_rag.ingest.staging",
    "scholar_rag.ingest.parse",
    "scholar_rag.ingest.metadata",
    "scholar_rag.ingest.clean",
    "scholar_rag.ingest.annotate",
    "scholar_rag.ingest.chunk",
    "scholar_rag.ingest.pipeline",
    "scholar_rag.store.layout",
    "scholar_rag.store.qdrant_manager",
    "scholar_rag.store.vector_store",
    "scholar_rag.store.catalog",
    "scholar_rag.retrieve.filters",
    "scholar_rag.retrieve.engine",
    "scholar_rag.services.mineru",
    "scholar_rag.services.crossref",
    "scholar_rag.services.pdf2doi",
    "scholar_rag.services.grobid",
    "scholar_rag.services.abstract_enrich",
    "scholar_rag.server.main",
    "scholar_rag.server.tools",
    "scholar_rag.server.schemas",
    "scholar_rag.cli",
    "scholar_rag.cli.setup",
]


@pytest.mark.parametrize("module", STUB_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_package_has_no_undocumented_modules():
    actual = sorted(m.name for m in pkgutil.iter_modules(scholar_rag.__path__))
    assert actual == ["cli", "core", "ingest", "models", "retrieve", "server", "services", "store"]


def test_server_registers_eleven_tools():
    from scholar_rag.server import tools

    assert list(tools.TOOL_NAMES) == [
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
    for tool_name in tools.TOOL_NAMES:
        assert callable(getattr(tools, f"tool_{tool_name}"))
