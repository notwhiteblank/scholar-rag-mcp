# Release Notes - v0.3.0

> Date: 2026-09-02. Major release removing the in-process local model backend; model access
> is now API-only through OpenAI-compatible servers (vLLM, Ollama, LM Studio, ...).

## New features

- **One-click install** (`uvx scholar-rag-mcp install`): installs the package via
  `uv tool install`, interactively resolves the six model-endpoint settings (defaults from
  Minimal environment unless a `SCHOLAR_RAG_*` variable or a `--chat-base-url` /
  `--chat-model` / `--embed-base-url` / `--embed-model` / `--rerank-base-url` /
  `--rerank-model` flag is given; `--yes` accepts all defaults), then registers the
  `scholar-rag-mcp` entry in the detected Claude Desktop / Claude Code / opencode / Codex
  user-level configs, touching only that entry. `--client <name>` restricts registration to
  one client; configs that cannot be parsed are left untouched and a manual snippet is
  printed.
- **Uninstall** (`scholar-rag-mcp uninstall`): removes the `scholar-rag-mcp` entry from the
  detected clients and runs `uv tool uninstall`; the knowledge-base data directory is kept.
  `--purge` deletes the data directory after explicit confirmation (or immediately with
  `--yes`).

## BREAKING CHANGES

- **Local model backend removed.** The in-process (transformers) chat/embed/rerank loaders
  are gone. Setting `SCHOLAR_RAG_CHAT_BACKEND=local` (also `SCHOLAR_RAG_EMBED_BACKEND` /
  `SCHOLAR_RAG_RERANK_BACKEND`) is now rejected at startup with a `ConfigError` that explains
  the removal and points to running models via an OpenAI-compatible server.
- **`local-models` pixi environment and pyproject extra removed.** The `local-models` feature
  environment and the `local-models` optional-dependencies extra are deleted; torch /
  transformers / accelerate are no longer project dependencies (MinerU keeps its own pinned
  stack).
- **Migration**: serve the models yourself (vLLM, Ollama, LM Studio, or any
  OpenAI-compatible endpoint), set each `*_MODEL` to the name that server exposes, and point
  `SCHOLAR_RAG_CHAT/EMBED/RERANK_BASE_URL` at it. See README "Model deployment".

## Improvements

- **`scripts/serve_models.sh` passes `--trust-remote-code`** to all three vLLM servers; it is
  required by the default model set (chat/embed/rerank) on the pinned vLLM (0.27.1). See
  `docs/spike-report-vllm-jina.md` for the spike findings.
- **Doctor no longer reports the `local-models` environment** as missing.

## Compatibility

Linux/Windows/macOS support is unchanged. The API-only model backend works with any
OpenAI-compatible server; the serve script remains Linux-only.

## Verified baseline

`pixi run lint && pixi run typecheck && pixi run test` are green.

## Upgrade from v0.2.1

Required: remove any `SCHOLAR_RAG_*_BACKEND=local` settings and configure `*_BASE_URL` for the
served models. Notes linked above (`docs/spike-report-vllm-jina.md`) exist only in the local
repository, not in the public `notwhiteblank/scholar-rag-mcp` mirror.