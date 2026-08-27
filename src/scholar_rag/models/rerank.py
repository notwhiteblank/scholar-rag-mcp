from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

DEFAULT_MODEL_NAME = "jina-reranker-v3.5"
_RERANKER_SCRIPT_NAME = "qwen3_vl_reranker"
_RERANKER_INSTRUCTION = "Determine if the document is relevant to the query."


class _ApiReranker(OpenAICompatClient):
    def rerank(self, query: str, documents: list[str], batch_size: int = 128) -> list[float]:
        if not documents:
            return []
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            response = self._post(
                "/rerank",
                {"model": self.model, "query": query, "documents": batch, "top_n": 0},
            )
            if response.status_code >= 400:
                raise ServiceUnavailableError(f"rerank failed: HTTP {response.status_code}")
            expected = len(batch)
            results = response.json().get("results", [])
            try:
                result_indices = [int(result["index"]) for result in results]
            except (KeyError, TypeError, ValueError):
                raise ServiceUnavailableError(
                    "rerank response contains a result with a missing or invalid index"
                ) from None
            if sorted(result_indices) != list(range(expected)):
                raise ServiceUnavailableError(
                    "rerank response index mismatch: expected "
                    f"{expected} scores covering indices 0..{expected - 1} exactly once, "
                    f"got {len(results)} results with indices {result_indices}"
                )
            chunk_scores = [0.0] * expected
            for result in results:
                chunk_scores[int(result["index"])] = float(result["relevance_score"])
            scores.extend(chunk_scores)
        return scores


class _LocalReranker:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model_path = str(Path(self._model_path).resolve())
        scripts_dir = Path(model_path) / "scripts"
        module_path = scripts_dir / f"{_RERANKER_SCRIPT_NAME}.py"
        if not module_path.is_file():
            raise ServiceUnavailableError(f"reranker script not found: {module_path}")
        torch: Any = importlib.import_module("torch")
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        spec = importlib.util.spec_from_file_location(_RERANKER_SCRIPT_NAME, module_path)
        if spec is None or spec.loader is None:
            raise ServiceUnavailableError(f"cannot load reranker script: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reranker_cls = module.Qwen3VLReranker
        self._model = reranker_cls(
            model_name_or_path=model_path,
            default_instruction=_RERANKER_INSTRUCTION,
            dtype=torch.float16,
        )

    def rerank(self, query: str, documents: list[str], batch_size: int = 128) -> list[float]:
        if not documents:
            return []
        self._ensure_loaded()
        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            inputs = {
                "instruction": _RERANKER_INSTRUCTION,
                "query": {"text": query},
                "documents": [{"text": doc} for doc in batch],
            }
            assert self._model is not None
            result = self._model.process(inputs)
            scores.extend(float(score) for score in result)
        return scores

    def health(self) -> bool:
        try:
            self._ensure_loaded()
        except Exception:
            return False
        return True

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            torch: Any = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class RerankClient:
    def __init__(
        self,
        *,
        backend: str = "api",
        base_url: str = "",
        api_key: str = "",
        model: str = DEFAULT_MODEL_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        self._api: _ApiReranker | None = None
        self._local: _LocalReranker | None = None
        if backend == "api":
            self._api = _ApiReranker(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self._local = _LocalReranker(model_path=model)

    def rerank(self, query: str, documents: list[str], batch_size: int = 128) -> list[float]:
        if self._api is not None:
            return self._api.rerank(query, documents, batch_size)
        assert self._local is not None
        return self._local.rerank(query, documents, batch_size)

    def health(self) -> bool:
        if self._api is not None:
            return self._api.health()
        assert self._local is not None
        return self._local.health()

    def close(self) -> None:
        if self._api is not None:
            self._api.close()
        elif self._local is not None:
            self._local.close()
