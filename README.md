# scholar-rag-mcp

<!-- mcp-name: io.github.notwhiteblank/scholar-rag-mcp -->

> **Status: preview release (v0.2.0).** Interfaces and storage layout may change in future versions.

**scholar-rag-mcp** is a publishable academic-paper knowledge-base MCP tool. Point it at a
folder of PDFs and it ingests each paper through a real parsing pipeline (MinerU), normalizes
metadata, annotates section structure, chunks and embeds the text, and stores everything in
Qdrant - after which an agent (or you) can semantically search chunks, run PubMed-style
document queries, read full text section by section, add/remove single papers, and manage
knowledge bases - all through 11 MCP tools over stdio. Embedding, annotation and re-ranking
run on OpenAI-compatible model services (e.g. vLLM) with in-process fallbacks.

## Features

- **Real ingestion pipeline**: MinerU PDF parsing (python/cli/api backends) -> metadata
  extraction (local heuristics, CrossRef, optional GROBID) -> cleaning -> section annotation
  -> deterministic chunking (configurable 300/1500/100 chars) -> embedding.
- **Fast retrieval at scale**: embedding first-pass + cross-encoder re-rank, optional
  metadata filtering (`doc_id`, `section`, `year`, `journal`, ...) evaluated inside the Qdrant
  index. 100k-chunk p95 query latency < 1s (see `docs/perf-report.md`).
- **Async jobs**: `create_kb`/`add_document` are background jobs with progress queryable via
  `get_job`; safe to restart (interrupted jobs are recovered and skipped on re-run).
- **Context-safe reading**: paginated `get_document_text` with hard size caps; *outline first,
  pages on demand*.
- **11 MCP tools** over stdio: `list_kbs`, `create_kb`, `delete_kb` (two-phase), `add_document`,
  `remove_document`, `get_document`, `get_document_text`, `list_documents`, `search_documents`,
  `search_chunks`, `get_job`.
- **Self-contained storage**: knowledge bases live under a single data directory
  (default location is platform-specific, see Data layout); Qdrant is either auto-launched
  (single binary, version-pinned) or connected to an external instance.

## Installation

Requires [pixi](https://pixi.sh). From the repository root:

scholar-rag-mcp runs on Linux x64, Windows x64 and macOS (Intel and Apple Silicon).
The pixi environments are locked for all four targets; the Qdrant binary used for
auto-launch is downloaded per platform on first use.

```bash
pixi install                      # installs the default environment
```

The project defines three pixi environments, each serving a different purpose:

| Environment | Purpose |
|---|---|
| `default` | Core runtime + dev tooling (pytest/ruff/mypy). Run the MCP server and all scripts here. |
| `mineru` | Adds MinerU (`==3.4.5`) plus its full runtime stack (pinned `transformers<5`, torch, onnxruntime, shapely, ...). Use for PDF parsing and the e2e smoke test. |
| `local-models` | Adds torch/transformers for in-process local model backends (falls back to downloading model weights on first use). |

Verify your environment with the built-in doctor:

```bash
pixi run python scripts/doctor.py
```

## Model deployment

Environment ('chat', 'embed' and 'rerank' clients) expects OpenAI-compatible HTTP endpoints.
`scripts/serve_models.sh` launches three vLLM instances for the reference model set:

| Service | Model | Port |
|---|---|---|
| chat | Qwen3.5-0.8B | 8101 |
| embed | jina-embeddings-v5-text-small | 8102 |
| rerank | jina-reranker-v3.5 | 8103 |

```bash
# point *_MODEL at your local model directories, then:
bash scripts/serve_models.sh
```

`SCHOLAR_RAG_CHAT_MODEL`, `SCHOLAR_RAG_EMBED_MODEL` and `SCHOLAR_RAG_RERANK_MODEL` are
**required** - the script exits with a message listing them if any is unset. Each value must
be an absolute path to a local HuggingFace model directory; vLLM serves each model under a
short name equal to the directory basename, so the client settings must use that short name
(the served name no longer equals the full path). Replace the `/path/to/...` placeholders in
`.env.example` accordingly. Ports (`CHAT_PORT`/`EMBED_PORT`/`RERANK_PORT`) and GPU ids remain
optional with working defaults.

The script pins the exact vLLM flags verified for these models (the Jina embed model needs
`--trust-remote-code` for its custom code; the reranker runs with its default task, no extra
flags). Model load takes several minutes; the script polls health until all three answer.

`scripts/serve_models.sh` is **Linux-only** (bash + CUDA + vLLM; vLLM has no Windows
support). On Windows/macOS point the `*_BASE_URL` settings at any OpenAI-compatible
server instead - for example Ollama (`http://127.0.0.1:11434/v1`), LM Studio's local
server, or a llama.cpp server - and set each `*_MODEL` to the model name that server
reports. The rerank endpoint must expose `/v1/rerank` (or leave reranking to the
embed-only fallback).

### Minimal environment

Start from `.env.example` and set at least the model endpoints (use the short names the
serve script exposes, equal to each model directory's basename):

```env
# data dir is optional - defaults to the platform data directory (see Data layout)
# SCHOLAR_RAG_DATA_DIR=
# SCHOLAR_RAG_QDRANT_STORAGE_DIR=

SCHOLAR_RAG_CHAT_BASE_URL=http://127.0.0.1:8101/v1
SCHOLAR_RAG_CHAT_MODEL=Qwen3.5-0.8B

SCHOLAR_RAG_EMBED_BASE_URL=http://127.0.0.1:8102/v1
SCHOLAR_RAG_EMBED_MODEL=jina-embeddings-v5-text-small

SCHOLAR_RAG_RERANK_BASE_URL=http://127.0.0.1:8103/v1
SCHOLAR_RAG_RERANK_MODEL=jina-reranker-v3.5
```

The embed model dimension is recorded in `kb_meta.json` at kb creation, so changing the
embedding model later requires a new kb.

## MCP client setup

Start the server entry point directly to make sure it runs:

```bash
pixi run scholar-rag-mcp
```

### Claude (Claude Desktop / claude CLI)

```json
{
  "mcpServers": {
    "scholar-rag-mcp": {
      "command": "pixi",
      "args": ["run", "scholar-rag-mcp"]
    }
  }
}
```

### opencode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "scholar-rag-mcp": {
      "type": "local",
      "command": ["pixi", "run", "scholar-rag-mcp"]
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `list_kbs` | List knowledge bases with document/chunk counts and status. |
| `create_kb` | Asynchronously ingest every PDF in a folder into a new kb (returns `job_id`). |
| `delete_kb` | Two-phase kb deletion (see below). |
| `add_document` | Asynchronously ingest a single PDF into an existing kb (returns `job_id`). |
| `remove_document` | Synchronously delete one document (Qdrant points + catalog + files). |
| `get_document` | Document overview: metadata, abstract, section outline, total size. |
| `get_document_text` | Paginated full-text reading of one document or a single section. |
| `list_documents` | Paginated browse of documents in a kb. |
| `search_documents` | PubMed-style document-level search (FTS + title/authors/journal/year). |
| `search_chunks` | Semantic chunk search with metadata filters and embed+rerank scores. |
| `get_job` | Query status/progress/result/elapsed time of a background job. |

## Data layout

```
<data_dir>/                     # SCHOLAR_RAG_DATA_DIR, default: platform data dir (below)
├── kbs/<kb_name>/
│   ├── kb_meta.json            # dimension, chunk config, schema version
│   ├── catalog.sqlite3         # documents / authors / keywords / chunks + FTS5
│   └── documents/<doc_id>/     # source.pdf, full_text.md, sections.json
├── cache/parse/                # MinerU markdown cache, keyed by content hash
├── cache/resolver/             # annotation resolver cache, keyed by content hash
├── jobs.sqlite3                # async job history
├── bin/                        # auto-downloaded Qdrant binary (v1.12.5)
└── qdrant-storage/             # default QDRANT_STORAGE_DIR location
```

Default `data_dir` per platform (override with `SCHOLAR_RAG_DATA_DIR`):

| Platform | Default |
|---|---|
| Linux | `$XDG_DATA_HOME/scholar-rag` (falls back to `~/.local/share/scholar-rag`) |
| macOS | `~/Library/Application Support/scholar-rag` |
| Windows | `%LOCALAPPDATA%\scholar-rag` |

Qdrant storage defaults to `<data_dir>/qdrant-storage` (override with
`SCHOLAR_RAG_QDRANT_STORAGE_DIR`) - it must be on a local filesystem, not a 9p/network
mount.

**Upgrading from v0.1.0 on Linux**: the old defaults (`~/.scholar-rag` and
`~/.local/share/scholar-rag/qdrant`) are migrated automatically on first start; if the
new location already has data, migration is skipped with a warning and the old files
are left untouched.

### Two-phase kb deletion

`delete_kb` never deletes on the first call with the wrong arguments by accident:

1. Call `delete_kb(kb="...")` - returns kb statistics plus a 10-minute `confirm_token`.
2. Call `delete_kb(kb="...", confirm_token="<token>")` to actually delete the Qdrant
   collection, kb directory and its job history.

## Development

```bash
pixi run lint          # ruff check src tests
pixi run typecheck     # mypy src
pixi run test          # pytest (unit + integration, no e2e/perf)
pixi run -e mineru pytest tests/e2e/smoke.py -v -m e2e   # real end-to-end smoke
python tests/perf/bench_query.py                          # query latency benchmark (writes docs/perf-report.md)
```

## Release notes

For known limitations and upgrade guidance see
`docs/handoffs/release-notes-v0.2.0.md`.

Known constraints worth repeating:

- **Qdrant is pinned to v1.12.5** - it is the highest version that runs on glibc 2.35;
  auto-launch downloads it on first use. On glibc >= 2.38 you may run a newer version, but the
  data format is not forward-compatible with older kbs in this release.
- **MinerU runs in its own pixi environment** because its transformers version is mutually
  exclusive with the vLLM one. PDF parsing thus prefers `pixi run -e mineru`.
- **MinerU weights** (~3.2 GB) download on first parse into `~/.cache/modelscope/`.
- **Metadata title heuristic**: titles are only picked locally when the MinerU markdown starts
  with an `#`/`##` heading, so a leading `## Abstract` (etc.) can be misread as the title. This
  affects the local-heuristic metadata tier only; the CrossRef tier (used when a DOI is found)
  normally corrects it.
- **Tool dispatch**: unknown extra arguments to a tool are silently ignored rather than
  rejected.
- **9p storage limit**: Qdrant storage must be on a local filesystem.
- **Platform support**: Linux x64 / Windows x64 / macOS Intel+Apple Silicon. The vLLM
  deployment script is Linux-only; Windows/macOS use any OpenAI-compatible server.
- **Windows long paths**: deep data directories can hit the 260-char limit; keep
  `SCHOLAR_RAG_DATA_DIR` shallow or enable Windows long path support.
- **macOS x64 CI**: covered at code level; CI matrix runs macOS arm64 (plus an x64
  runner when available).