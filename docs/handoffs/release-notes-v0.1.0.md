# Release Notes - v0.1.0

> Date: 2026-08-26. This release is the first publishable milestone of
> `scholar-rag-mcp`, covering the full ingest -> store -> retrieve -> MCP toolchain.

## What's in v0.1.0

- **Config & runtime** (`core/`): pydantic-settings based `SCHOLAR_RAG_*` configuration
  (env vars or `<data_dir>/config.json`), structured error model with stable error codes,
  SQLite-backed async job system with interruption recovery, KB registry with per-kb write
  locks, two-phase (token-based) kb deletion.
- **Model clients** (`models/`): OpenAI-compatible chat/embedding/rerank clients with a
  local in-process fallback per client; lazy singletons with `health()`.
- **Storage** (`store/`): Qdrant-backed vector store (auto-launch of the pinned v1.12.5
  binary or external instance), per-kb collections with keyword/integer payload indexes,
  deterministic UUID5 point ids, sqlite catalog with FTS5.
- **Ingestion** (`ingest/`): staging (content-hash dedup) -> MinerU parse (python/cli/api)
  -> metadata (local heuristics / CrossRef / optional GROBID) -> clean -> section
  annotation -> deterministic chunking -> embed -> three-way persist (qdrant + catalog +
  files). Per-document error isolation with job-level reports and `skip_existing` resume.
- **Retrieval** (`retrieve/`): embedding first pass (<=200 candidates) + cross-encoder
  re-rank, whitelisted metadata filters executed inside the Qdrant index, graceful
  degradation to embed-only scoring when the reranker is unavailable.
- **MCP server** (`server/`): 11-tool stdio server with validated argument schemas,
  structured `{error_code, message, hint}` errors, context-safe pagination caps, and
  `get_job` elapsed-time reporting.
- **Docs & tooling**: `scripts/doctor.py` environment self-check, `tests/perf/bench_query.py`
  latency benchmark with real (not synthetic) result tables in `docs/perf-report.md`, a
  full-config `.env.example`, and a real end-to-end smoke test in `tests/e2e/smoke.py`.

## Verified baseline

`pixi run lint && pixi run typecheck && pixi run test` are green (ruff, mypy, pytest).
The 100k-chunk query benchmark passes its SPEC §11 budget (unfiltered and filtered
`p95 < 1s`); see `docs/perf-report.md` for the measured numbers.

## Known limitations

1. **MinerU environment separation** - MinerU requires transformers < 5, which conflicts
   with the vLLM-serving environment. PDF parsing must therefore run in the `mineru` pixi
   environment (`pixi run -e mineru ...`). The MCP server does not parse PDFs in-process by
   default; deploy `add_document`/`create_kb` from a process that has MinerU available.
2. **Qdrant pinned to v1.12.5** - needed for glibc 2.35 hosts; auto-launch downloads it.
   Newer Qdrant data formats are not read by this release.
3. **Legacy kb incompatibility** - v0.1.0 does not read knowledge bases produced by the
   pre-release snapshot (different collection naming, payload schema and modeling). Old kbs
   must be re-ingested; the storage layout (`data_dir/kbs/`) coexists on disk without
   touching legacy data.
4. **Embedding model is kb-bound** - the embedding dimension is recorded in `kb_meta.json`
   at kb creation. Switching embedding models requires creating a new kb.
5. **9p / network-mount storage** - Qdrant storage must reside on a local filesystem
   (`QDRANT_STORAGE_DIR`); 9p mounts cause `EINVAL`.
6. **Single-process job concurrency** - `JOB_WORKERS` defaults to 1 to protect GPU memory;
   raising it increases model-service load.
7. **Cross-encoder re-rank on API**: rerank latency is bounded by the fixed candidate cap
   (<=200) and batching (<=128 per request).

## Upgrade / migration notes

- **From v0.1.0-rc (git history) without this release**: no persistent format changed
  during the P-series; old kbs created before this tag remain valid. Only `get_job` gained a
  `timings.elapsed_s` field in the output schema - clients that strict-validated
  `timings == {}` must accept an `elapsed_s` float now.
- **Fresh installs** should start from `.env.example`; keep `QDRANT_STORAGE_DIR` on local
  disk and point the three `*_BASE_URL`/`*_MODEL` pairs at your model services.
- **If MinerU is unavailable** in your runtime env, run ingestion from the `mineru` pixi
  environment and point `add_document`/`create_kb` at PDFs there.
- Run `scripts/doctor.py` after any environment change to confirm data dir, model services,
  MinerU and Qdrant are all healthy.

## What is NOT in v0.1.0

- No remote (HTTP/SSE) MCP transport; stdio only.
- No incremental re-indexing or kb merge/split.
- No bulk import beyond a folder scan on `create_kb`.
- No authenticated model services (expects trusted local vLLM endpoints).