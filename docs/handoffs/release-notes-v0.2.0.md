# Release Notes - v0.2.0

> Date: 2026-08-31. This release makes `scholar-rag-mcp` fully usable on
> Windows x64, macOS x64 (Intel) and macOS arm64 (Apple Silicon), with Linux x64
> behavior unchanged.

## What's new in v0.2.0

- **Cross-platform support**: the project runs on Linux x64, Windows x64 and
  macOS (Intel and Apple Silicon). The Qdrant v1.12.5 binary is downloaded per
  platform on first use (Linux tar.gz, macOS tar.gz, Windows zip with
  `qdrant.exe`); auto-launch works on all four targets.
- **Platform-convention data directories**: the default data directory now
  follows each platform's convention instead of a fixed Linux path -
  `$XDG_DATA_HOME/scholar-rag` on Linux, `~/Library/Application Support/scholar-rag`
  on macOS, `%LOCALAPPDATA%\scholar-rag` on Windows. Override with
  `SCHOLAR_RAG_DATA_DIR`; Qdrant storage defaults to `<data_dir>/qdrant-storage`
  (override with `SCHOLAR_RAG_QDRANT_STORAGE_DIR`).
- **Cross-platform CI** (Phase 5): the GitHub Actions matrix runs the test suite
  on Ubuntu, Windows and macOS runners; macOS arm64 is covered by
  `macos-latest`. macOS x64 is supported at code level (no x64 macOS CI runner:
  GitHub x64 macOS runners require larger-runner billing).
- **Documentation**: README and `.env.example` now document platform defaults,
  the Linux-only deployment script, and per-platform constraints.

> **Breaking change**: on Linux only, the old default data directory
> (`~/.scholar-rag`, with Qdrant storage at `~/.local/share/scholar-rag/qdrant`)
> is **migrated automatically** to the new XDG-based defaults on first start.
> Migration updates your existing data in place; failure to migrate aborts
> startup with a clear message. Do NOT set any variable and simply start the
> server to let migration happen. Users who explicitly set
> `SCHOLAR_RAG_DATA_DIR` (or `SCHOLAR_RAG_QDRANT_STORAGE_DIR`) are **unaffected**.
> Before this release, the default data directory was fixed at `~/.scholar-rag`;
> any configuration or scripts that relied on that location should be reviewed.
> **WARNING**: if your previous `~/.scholar-rag` is on a 9p/network mount, the
> migration (or the new `<data_dir>` location) will fail for the same reason.

## Verified baseline

`pixi run lint && pixi run typecheck && pixi run test` are green (ruff, mypy,
pytest). The 100k-chunk query benchmark continues to pass its SPEC §11 budget
(`p95 < 1s`); see `docs/perf-report.md`.

## Known limitations

1. **vLLM deployment script is Linux-only** - `scripts/serve_models.sh` requires
   bash + CUDA + vLLM, and vLLM has no Windows support. On Windows/macOS point
   the `*_BASE_URL` settings at any OpenAI-compatible server (e.g. Ollama at
   `http://127.0.0.1:11434/v1`, LM Studio's local server, or a llama.cpp server)
   and set each `*_MODEL` to the model name that server reports. The rerank
   endpoint must expose `/v1/rerank` (or leave reranking to the embed-only
   fallback).
2. **macOS x64 (Intel)** - covered at code level; macOS x64 wheels are no longer
   published for the torch ecosystem, so the `mineru`/`local-models` pixi
   environments are not available on osx-64 (the default environment is). CI
   covers macOS arm64 via `macos-latest`; there is no x64 macOS CI runner
   (GitHub x64 macOS runners require larger-runner billing).
3. **Windows long paths** - deep data directories can hit the 260-char path
   limit; keep `SCHOLAR_RAG_DATA_DIR` shallow or enable Windows long path
   support.
4. **9p / network-mount storage** - Qdrant storage must reside on a local
   filesystem (`SCHOLAR_RAG_QDRANT_STORAGE_DIR`); 9p mounts cause `EINVAL`.

## Upgrade from v0.1.0

- **No action needed**: v0.1.0 users simply start the new version; on first
  start the Linux legacy data directory is migrated automatically (see the
  breaking-change note above). If the new location already contains data,
  migration is skipped with a warning and the old files are left untouched.
- Fresh installs should start from `.env.example`; keep Qdrant storage on local
  disk and point the three `*_BASE_URL`/`*_MODEL` pairs at your model services.
- Run `scripts/doctor.py` after any environment change to confirm data dir,
  model services, MinerU and Qdrant are all healthy.

## What is NOT in v0.2.0

- No remote (HTTP/SSE) MCP transport; stdio only.
- No incremental re-indexing or kb merge/split.
- No bulk import beyond a folder scan on `create_kb`.
- No authenticated model services (expects trusted local OpenAI-compatible
  endpoints).