# scholar-rag-mcp

A run-in-place MCP tool that turns your local academic paper corpus into a searchable, chat-capable knowledge base (ingest with MinerU, store with Qdrant, embed/rerank via vLLM-compatible APIs).

## Development

Requires [pixi](https://pixi.sh).

```
pixi run lint
pixi run typecheck
pixi run test
```

See `docs/handoffs/p2-handoff.md` for environment and configuration details.