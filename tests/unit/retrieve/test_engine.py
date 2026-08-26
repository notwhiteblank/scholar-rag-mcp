import pytest
from qdrant_client import models

from scholar_rag.retrieve import engine as engine_module
from scholar_rag.retrieve.engine import RetrievalEngine


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.vector = [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return list(self.vector)


class FakeReranker:
    def __init__(self) -> None:
        self.scores: list[float] | None = None
        self.error: Exception | None = None
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.queries.append(query)
        self.documents.append(list(documents))
        if self.error is not None:
            raise self.error
        if self.scores is not None:
            return list(self.scores)
        return [1.0 for _ in documents]


class FakeStore:
    registry: dict = {}
    last: "FakeStore | None" = None

    @classmethod
    def open(cls, kb: str) -> "FakeStore":
        store = cls(cls.registry[kb])
        cls.last = store
        return store

    def __init__(self, points) -> None:
        self._order = [point_id for point_id, _score, _payload in points]
        self._scores = {point_id: score for point_id, score, _payload in points}
        self._payloads = {point_id: _payload for point_id, _score, _payload in points}
        self.query_limit: int | None = None
        self.query_filter = None

    def query(self, vector, qfilter, limit: int):
        self.query_limit = limit
        self.query_filter = qfilter
        return [(point_id, self._scores[point_id]) for point_id in self._order]

    def payloads(self, point_ids: list[str]):
        return {point_id: self._payloads[point_id] for point_id in point_ids if point_id in self._payloads}


def _points(count: int, prefix: str = "c") -> list:
    return [
        (f"{prefix}{index}", 1.0 - index * 0.01, {"content": f"{prefix}{index}", "doc_id": prefix})
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def _reset_store():
    FakeStore.registry = {}
    FakeStore.last = None


@pytest.fixture
def harness(monkeypatch):
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    monkeypatch.setattr(engine_module, "get_embedding_client", lambda: embedder)
    monkeypatch.setattr(engine_module, "get_rerank_client", lambda: reranker)
    monkeypatch.setattr(engine_module, "VectorStore", FakeStore)
    return embedder, reranker


@pytest.mark.parametrize(("top_k", "expected"), [(10, 50), (50, 200), (1, 5)])
def test_candidate_limit(harness, top_k, expected):
    FakeStore.registry["kb"] = _points(60)
    RetrievalEngine("kb").search("q", top_k=top_k)
    assert FakeStore.last is not None
    assert FakeStore.last.query_limit == expected


def test_rerank_scores_and_order(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("b", 0.8, {"content": "b"}), ("c", 0.7, {"content": "c"}), ("a", 0.6, {"content": "a"})]
    reranker.scores = [0.2, 0.9, 0.5]
    outcome = RetrievalEngine("kb").search("q")
    assert [chunk.content for chunk in outcome.chunks] == ["c", "a", "b"]
    assert [chunk.rerank_score for chunk in outcome.chunks] == [0.9, 0.5, 0.2]
    assert [chunk.embed_score for chunk in outcome.chunks] == [0.7, 0.6, 0.8]
    assert outcome.warning is None
    assert reranker.documents == [["b", "c", "a"]]


def test_chunk_content_and_full_metadata(harness):
    payload = {"content": "text body", "doc_id": "d", "year": 2024}
    FakeStore.registry["kb"] = [("id1", 0.9, payload)]
    outcome = RetrievalEngine("kb").search("q")
    assert len(outcome.chunks) == 1
    chunk = outcome.chunks[0]
    assert chunk.content == "text body"
    assert chunk.metadata == payload
    assert chunk.embed_score == 0.9


def test_rerank_exception_degrades_to_embed_order(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("b", 0.8, {"content": "b"}), ("c", 0.7, {"content": "c"}), ("a", 0.6, {"content": "a"})]
    reranker.error = RuntimeError("boom")
    outcome = RetrievalEngine("kb").search("q")
    assert [chunk.content for chunk in outcome.chunks] == ["b", "c", "a"]
    assert [chunk.embed_score for chunk in outcome.chunks] == [0.8, 0.7, 0.6]
    assert all(chunk.rerank_score is None for chunk in outcome.chunks)
    assert outcome.warning == "reranker_unavailable"


def test_min_score_normal_keeps_equal_threshold(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("a", 0.5, {"content": "a"}), ("b", 0.4, {"content": "b"}), ("c", 0.3, {"content": "c"})]
    reranker.scores = [0.9, 0.8, 0.6]
    outcome = RetrievalEngine("kb").search("q", min_score=0.8)
    assert [chunk.content for chunk in outcome.chunks] == ["a", "b"]
    assert [chunk.rerank_score for chunk in outcome.chunks] == [0.9, 0.8]


def test_min_score_degraded_applies_to_embed_score(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("a", 0.95, {"content": "a"}), ("b", 0.9, {"content": "b"}), ("c", 0.85, {"content": "c"})]
    reranker.error = RuntimeError("boom")
    outcome = RetrievalEngine("kb").search("q", min_score=0.9)
    assert [chunk.content for chunk in outcome.chunks] == ["a", "b"]
    assert all(chunk.rerank_score is None for chunk in outcome.chunks)


def test_top_k_truncation(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("a", 0.5, {"content": "a"}), ("b", 0.4, {"content": "b"}), ("c", 0.3, {"content": "c"})]
    reranker.scores = [0.1, 0.9, 0.5]
    outcome = RetrievalEngine("kb").search("q", top_k=2)
    assert [chunk.content for chunk in outcome.chunks] == ["b", "c"]


def test_empty_candidates_returns_empty_with_timings(harness):
    FakeStore.registry["kb"] = []
    outcome = RetrievalEngine("kb").search("q")
    assert outcome.chunks == []
    assert list(outcome.timings) == ["embed_ms", "search_ms", "rerank_ms"]
    assert outcome.timings["rerank_ms"] == 0.0
    assert all(outcome.timings[key] >= 0.0 for key in outcome.timings)


def test_timings_keys_present_in_normal_path(harness):
    FakeStore.registry["kb"] = [("a", 0.9, {"content": "a"})]
    outcome = RetrievalEngine("kb").search("q")
    assert set(outcome.timings) == {"embed_ms", "search_ms", "rerank_ms"}
    assert all(outcome.timings[key] >= 0.0 for key in outcome.timings)


def test_degraded_path_timings_keys_present(harness):
    _, reranker = harness
    FakeStore.registry["kb"] = [("a", 0.9, {"content": "a"})]
    reranker.error = RuntimeError("boom")
    outcome = RetrievalEngine("kb").search("q")
    assert set(outcome.timings) == {"embed_ms", "search_ms", "rerank_ms"}
    assert all(outcome.timings[key] >= 0.0 for key in outcome.timings)


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": 51}, {"min_score": -0.1}, {"min_score": 1.1}])
def test_out_of_range_args_raise_value_error(harness, kwargs):
    FakeStore.registry["kb"] = _points(2)
    with pytest.raises(ValueError):
        RetrievalEngine("kb").search("q", **kwargs)


def test_embed_query_called_with_query_text(harness):
    embedder, _ = harness
    FakeStore.registry["kb"] = [("a", 0.9, {"content": "a"})]
    RetrievalEngine("kb").search("my query")
    assert embedder.queries == ["my query"]


def test_metadata_filter_translated_and_passed_to_store(harness):
    FakeStore.registry["kb"] = _points(3)
    RetrievalEngine("kb").search("q", metadata_filter={"journal": "TMLR"})
    assert FakeStore.last is not None
    assert isinstance(FakeStore.last.query_filter, models.Filter)
