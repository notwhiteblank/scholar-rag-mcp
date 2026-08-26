from __future__ import annotations

import gc
import importlib.util
import sys
from pathlib import Path
from typing import Any

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

DEFAULT_MODEL_NAME = "Qwen3-VL-Embedding-2B"
_EMBEDDING_SCRIPT_NAME = "qwen3_vl_embedding"


class _ApiEmbeddings(OpenAICompatClient):
    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._post("/embeddings", {"model": self.model, "input": batch})
            if response.status_code >= 400:
                raise ServiceUnavailableError(
                    f"embedding failed: HTTP {response.status_code}"
                )
            expected = len(batch)
            items = response.json().get("data", [])
            if len(items) != expected:
                raise ServiceUnavailableError(
                    "embedding response length mismatch: "
                    f"expected {expected} vectors, got {len(items)}"
                )
            try:
                indices = [int(item.get("index", -1)) for item in items]
            except (KeyError, TypeError, ValueError):
                raise ServiceUnavailableError(
                    "embedding response contains an item with an invalid index"
                ) from None
            if sorted(indices) != list(range(expected)):
                raise ServiceUnavailableError(
                    "embedding response index mismatch: expected indices "
                    f"0..{expected - 1} exactly once, got indices {indices}"
                )
            ordered = sorted(items, key=lambda item: int(item.get("index", -1)))
            for item in ordered:
                all_embeddings.append(list(item["embedding"]))
        return all_embeddings


class _LocalEmbeddings:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        model_path = str(Path(self._model_path).resolve())
        scripts_dir = Path(model_path) / "scripts"
        module_path = scripts_dir / f"{_EMBEDDING_SCRIPT_NAME}.py"
        if not module_path.is_file():
            raise ServiceUnavailableError(f"embedding script not found: {module_path}")
        torch: Any = importlib.import_module("torch")
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        spec = importlib.util.spec_from_file_location(_EMBEDDING_SCRIPT_NAME, module_path)
        if spec is None or spec.loader is None:
            raise ServiceUnavailableError(f"cannot load embedding script: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        embedder_cls = module.Qwen3VLEmbedder
        self._model = embedder_cls(model_name_or_path=model_path, dtype=torch.float16)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = [{"text": text} for text in batch]
            assert self._model is not None
            embeddings = self._model.process(inputs, normalize=True).cpu().float().tolist()
            all_embeddings.extend(embeddings)
        return all_embeddings

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


class EmbeddingClient:
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
        self._api: _ApiEmbeddings | None = None
        self._local: _LocalEmbeddings | None = None
        if backend == "api":
            self._api = _ApiEmbeddings(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self._local = _LocalEmbeddings(model_path=model)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        if self._api is not None:
            return self._api.embed(texts, batch_size)
        assert self._local is not None
        return self._local.embed(texts, batch_size)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

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
