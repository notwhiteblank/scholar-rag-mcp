#!/usr/bin/env python3
"""Query-performance benchmark for the retrieval engine.

Builds a synthetic 100,000-chunk collection (2000 documents x 50 chunks) in a
throwaway real Qdrant instance (storage on local disk), then measures p50/p95/max
latency of the full RetrievalEngine path for unfiltered and filtered searches and
asserts the SPEC section 11 budget: p95 < 1s for both, with at least 50 queries each.

Embeddings default to deterministic fake vectors (fixed seed). Pass --real-embed to
route query embedding through a real HTTP embedding service configured via the
SCHOLAR_RAG_EMBED_* environment variables.

The report is written to docs/perf-report.md.

Usage:
    python tests/perf/bench_query.py [--real-embed] [--queries N]
"""
from __future__ import annotations

import argparse
import os
import random
import tempfile
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import models

import scholar_rag.retrieve.engine as engine_module
from scholar_rag.retrieve.engine import RetrievalEngine
from scholar_rag.store.qdrant_manager import get_qdrant_manager
from scholar_rag.store.vector_store import VectorStore, _point_id

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPORT_PATH = _REPO_ROOT / "docs" / "perf-report.md"
_SPIKE_BINARY = _REPO_ROOT / "spike" / "out" / "qdrant" / (
    "qdrant.exe" if os.name == "nt" else "qdrant"
)
_DIM = 2048
_DOCS = 2000
_CHUNKS_PER_DOC = 50
_BATCH = 256
_P95_TARGET_MS = 1000.0
_MIN_QUERIES = 50
_TOTAL_CHUNKS = _DOCS * _CHUNKS_PER_DOC

_PARA_TEMPLATE = (
    "Synthetic academic paragraph {} discussing method {}. "
    "Measurements were collected and averaged over repeated trials, and the results "
    "were cross-validated against the reference implementation on a multi-year benchmark."
)
_JOURNALS = ["Journal of Topics 0", "Journal of Topics 1", "Journal of Topics 2",
             "Journal of Topics 3", "Journal of Topics 4", "Journal of Topics 5",
             "Journal of Topics 6"]


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_env(real_embed: bool) -> tuple[Path, Path]:
    """Point SCHOLAR_RAG_* at throwaway local paths and return (root, data_dir)."""
    root = Path(tempfile.mkdtemp(prefix="scholar-bench-"))
    storage = root / "qdrant"
    data_dir = root / "data"
    os.environ["SCHOLAR_RAG_DATA_DIR"] = str(data_dir)
    os.environ["SCHOLAR_RAG_QDRANT_STORAGE_DIR"] = str(storage)
    if _SPIKE_BINARY.is_file():
        os.environ["SCHOLAR_RAG_QDRANT_BIN"] = str(_SPIKE_BINARY)
    if not real_embed:
        for key in ("SCHOLAR_RAG_EMBED_BASE_URL", "SCHOLAR_RAG_EMBED_MODEL",
                    "SCHOLAR_RAG_EMBED_API_KEY"):
            os.environ.pop(key, None)
    return root, data_dir


def make_payload(doc_index: int, chunk_index: int) -> dict[str, object]:
    year = 1990 + (doc_index % 35)
    journal = _JOURNALS[doc_index % len(_JOURNALS)]
    first_author = f"Author {doc_index % 300}"
    return {
        "doc_id": f"doc-{doc_index:05d}",
        "chunk_index": chunk_index,
        "title": f"Synthetic paper {doc_index} on broad topics",
        "title_norm": f"synthetic paper {doc_index} on broad topics",
        "year": year,
        "journal": journal,
        "journal_norm": journal.lower(),
        "first_author": first_author,
        "first_author_norm": first_author.lower(),
        "author_count": (doc_index % 5) + 1,
        "doi": f"10.1000/synthetic.{doc_index:05d}",
        "source_file": "not_applicable/benchmark",
        "content_hash": f"bench:{doc_index:05d}#{chunk_index}",
        "added_at": _now_iso(),
        "added_ts": 1704067200.0,
        "section": "Body" if chunk_index % 4 else "Intro",
        "heading_path": f"Paper {doc_index} / Body {chunk_index // 4}",
        "content": _PARA_TEMPLATE.format(chunk_index, (doc_index + chunk_index) % 40)
        + f" The abstract claims a relative improvement of {20 + (doc_index + chunk_index) % 60} percent.",
    }


def _random_unit_vector(seq: random.Random, dim: int) -> list[float]:
    vector = [seq.uniform(0.0, 1.0) for _ in range(dim)]
    norm = sum(v * v for v in vector) ** 0.5
    return [v / norm for v in vector]


def build_collection(docs: int, per_doc: int) -> int:
    """Populate a fresh kb_bench collection with docs*per_doc synthetic points."""
    client = get_qdrant_manager().client()
    store = VectorStore.create("bench", _DIM)
    seq = random.Random(20260826)
    total = 0
    points: list[models.PointStruct] = []
    for d in range(docs):
        for c in range(per_doc):
            payload = make_payload(d, c)
            points.append(
                models.PointStruct(
                    id=_point_id(str(payload["doc_id"]), c),
                    vector=_random_unit_vector(seq, _DIM),
                    payload=payload,
                )
            )
            if len(points) >= _BATCH:
                client.upsert(collection_name=store._collection, points=points, wait=False)
                total += len(points)
                points = []
    if points:
        client.upsert(collection_name=store._collection, points=points, wait=False)
        total += len(points)
    return total


def install_fakes(real_embed: bool) -> None:
    if not real_embed:
        class _FakeEmbedder:
            def embed_query(self, text: str) -> list[float]:
                return _random_unit_vector(random.Random(zlib.crc32(text.encode())), _DIM)

            def embed(self, texts: list[str]) -> list[list[float]]:
                return [_random_unit_vector(random.Random(zlib.crc32(t.encode())), _DIM)
                        for t in texts]

        engine_module.get_embedding_client = lambda: _FakeEmbedder()

    class _FakeReranker:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [round(0.5 - index * 0.0001, 4) for index in range(len(documents))]

    engine_module.get_rerank_client = lambda: _FakeReranker()


def wait_ready(expected: int) -> None:
    client = get_qdrant_manager().client()
    deadline = time.monotonic() + 600
    count = 0
    while time.monotonic() < deadline:
        count = client.count(collection_name="kb_bench").count
        if count >= expected:
            break
        time.sleep(1)
    if count < expected:
        raise RuntimeError(f"collection never reached {expected} points (last count {count})")


def measure(queries: list[tuple[str, dict | None]]) -> list[float]:
    engine = RetrievalEngine("bench")
    latencies: list[float] = []
    for query, metadata_filter in queries:
        start = time.perf_counter()
        engine.search(query, top_k=10, metadata_filter=metadata_filter)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return latencies


def percentile(sorted_values: list[float], kth: float) -> float:
    index = min(len(sorted_values) - 1, int(round(kth / 100.0 * (len(sorted_values) - 1))))
    return sorted_values[index]


def summarize(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return percentile(ordered, 50), percentile(ordered, 95), max(ordered)


def environment_lines() -> list[str]:
    import platform

    lines = [
        f"| platform | {platform.system()} {platform.machine()} |",
        "| qdrant | real v1.12.5 binary, local-filesystem storage |",
        f"| collection | {_TOTAL_CHUNKS:,} points, dim={_DIM}, cosine |",
    ]
    if os.environ.get("BENCH_REAL_EMBED") == "1":
        lines.append(
            f"| embedding | real HTTP via {os.environ.get('SCHOLAR_RAG_EMBED_BASE_URL', '')} |"
        )
    else:
        lines.append("| embedding | fake deterministic (seeded random) |")
    lines.append("| rerank | fake deterministic |")
    return lines


def write_report(rows: list[tuple[str, int, float, float, float, str]], overall: str) -> None:
    lines = [
        "# Performance Report - Query Benchmark (v0.1.0)",
        "",
        "> Auto-generated by `tests/perf/bench_query.py`. Do not edit by hand.",
        "",
        "## Method",
        "",
        f"- Synthetic collection: {_DOCS} documents x {_CHUNKS_PER_DOC} chunks = "
        f"{_TOTAL_CHUNKS:,} chunks, 2048-dim deterministic vectors.",
        "- Real Qdrant v1.12.5 started on temporary local-filesystem storage; full "
        "`VectorStore`/`RetrievalEngine` path (embed -> indexed filter search -> payload "
        "fetch -> rerank).",
        "- Budget (SPEC section 11): unfiltered and filtered query p95 < 1s.",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    lines += environment_lines()
    lines += [
        "",
        "## Results (latency in ms)",
        "",
        "| Query class | queries | p50 (ms) | p95 (ms) | max (ms) | budget | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, count, p50, p95, maximum, verdict in rows:
        lines.append(
            f"| {label} | {count} | {p50:.1f} | {p95:.1f} | {maximum:.1f} | p95<1000 | {verdict} |"
        )
    lines += [
        "",
        f"**Overall verdict: {overall}**",
        "",
    ]
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written to {_REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-embed", action="store_true",
                        help="route query embedding through the configured SCHOLAR_RAG_EMBED_* "
                        "HTTP service instead of fake vectors")
    parser.add_argument("--queries", type=int, default=100,
                        help="number of queries per class (default 100)")
    parser.add_argument("--docs", type=int, default=_DOCS,
                        help="number of synthetic documents (default 2000)")
    parser.add_argument("--chunks", type=int, default=_CHUNKS_PER_DOC,
                        help="chunks per document (default 50)")
    args = parser.parse_args(argv)
    if args.queries < _MIN_QUERIES:
        parser.error(f"--queries must be >= {_MIN_QUERIES}")
    if args.real_embed and not os.environ.get("SCHOLAR_RAG_EMBED_BASE_URL"):
        parser.error(
            "--real-embed requires SCHOLAR_RAG_EMBED_BASE_URL to be set to an "
            "OpenAI-compatible embeddings endpoint (e.g. http://127.0.0.1:8102/v1); "
            "optionally configure SCHOLAR_RAG_EMBED_MODEL and SCHOLAR_RAG_EMBED_API_KEY"
        )
    if args.real_embed:
        os.environ["BENCH_REAL_EMBED"] = "1"

    root, _data_dir = setup_env(args.real_embed)
    print(f"bench root: {root}")
    print("building collection...")
    total = build_collection(args.docs, args.chunks)
    print(f"upserted {total:,} points")
    wait_ready(total)
    print("collection ready")

    install_fakes(args.real_embed)

    filtered_docs = [f"doc-{i:05d}" for i in range(1000)]
    unfiltered = [(f"query about method {i} results", None) for i in range(args.queries)]
    filtered = [
        (f"filtered query on method {i}",
         {"doc_id": filtered_docs, "year": {"gte": 1995, "lte": 2020}})
        for i in range(args.queries)
    ]

    rows: list[tuple[str, int, float, float, float, str]] = []
    for label, queries in (("unfiltered", unfiltered), ("filtered", filtered)):
        latencies = measure(queries)
        p50, p95, maximum = summarize(latencies)
        verdict = "PASS" if p95 < _P95_TARGET_MS else "FAIL"
        rows.append((label, len(latencies), p50, p95, maximum, verdict))
        print(f"{label}: {len(latencies)} queries p50={p50:.1f}ms p95={p95:.1f}ms max={maximum:.1f}ms")

    overall = "PASS" if all(row[-1] == "PASS" for row in rows) else "FAIL"
    write_report(rows, overall)
    if overall == "FAIL":
        print("BUDGET FAILED: p95 >= 1s for one or more query classes")
        return 1
    print("BUDGET PASSED (p95 < 1s for all query classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
