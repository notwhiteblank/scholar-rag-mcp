from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from scholar_rag.core.errors import ConfigError

ENV_PREFIX = "SCHOLAR_RAG_"
_MODEL_BACKENDS = frozenset({"api", "local"})
_MINERU_BACKENDS = frozenset({"python", "cli", "api"})
_DEFAULT_DATA_DIR = "~/.scholar-rag"
_DEFAULT_QDRANT_STORAGE_DIR = "~/.local/share/scholar-rag/qdrant"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    data_dir: Path = Field(default_factory=lambda: Path(_DEFAULT_DATA_DIR).expanduser())
    qdrant_url: str = ""
    qdrant_bin: str = ""
    qdrant_port: int = 6333
    qdrant_storage_dir: Path = Field(
        default_factory=lambda: Path(_DEFAULT_QDRANT_STORAGE_DIR).expanduser()
    )
    chat_backend: str = "api"
    chat_base_url: str = ""
    chat_api_key: str = ""
    chat_model: str = "Qwen3.5-0.8B"
    embed_backend: str = "api"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "jina-embeddings-v5-text-small"
    rerank_backend: str = "api"
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "jina-reranker-v3.5"
    mineru_backend: str = "python"
    mineru_api_url: str = "http://127.0.0.1:8010"
    mineru_bin: str = "mineru"
    keywords_enabled: bool = True
    annotation_resolver_enabled: bool = True
    job_workers: int = 1
    crossref_enabled: bool = True
    grobid_url: str = ""
    log_level: str = "WARNING"
    log_dir: str = ""

    @field_validator("data_dir", "qdrant_storage_dir", mode="before")
    @classmethod
    def _expand_home(cls, value: object) -> object:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @field_validator("chat_backend", "embed_backend", "rerank_backend")
    @classmethod
    def _validate_model_backend(cls, value: str, info: ValidationInfo) -> str:
        if value not in _MODEL_BACKENDS:
            allowed = ", ".join(sorted(_MODEL_BACKENDS))
            raise ConfigError(f"invalid {info.field_name}: {value!r} (allowed: {allowed})")
        return value

    @field_validator("mineru_backend")
    @classmethod
    def _validate_mineru_backend(cls, value: str) -> str:
        if value not in _MINERU_BACKENDS:
            allowed = ", ".join(sorted(_MINERU_BACKENDS))
            raise ConfigError(f"invalid mineru_backend: {value!r} (allowed: {allowed})")
        return value

    @classmethod
    def load(cls) -> Settings:
        values = cls._read_config_file()
        values.update(cls._read_environment())
        return cls(**values)

    @classmethod
    def _read_config_file(cls) -> dict[str, Any]:
        path = cls._config_file_path()
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"failed to read config file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config file {path} must contain a JSON object")
        values: dict[str, Any] = {}
        for key, value in raw.items():
            field_name = cls._match_field_name(key)
            if field_name is not None:
                values[field_name] = value
        return values

    @classmethod
    def _read_environment(cls) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(ENV_PREFIX):
                continue
            field_name = key[len(ENV_PREFIX) :].lower()
            if field_name in cls.model_fields:
                values[field_name] = value
        return values

    @classmethod
    def _config_file_path(cls) -> Path:
        env_dir = os.environ.get(f"{ENV_PREFIX}DATA_DIR")
        base = Path(env_dir or _DEFAULT_DATA_DIR).expanduser()
        return base / "config.json"

    @classmethod
    def _match_field_name(cls, key: str) -> str | None:
        name = key[len(ENV_PREFIX) :] if key.startswith(ENV_PREFIX) else key
        normalized = name.upper()
        for field_name in cls.model_fields:
            if field_name.upper() == normalized:
                return field_name
        return None

    def validate_runtime(self) -> None:
        missing: list[str] = []
        for group in ("chat", "embed", "rerank"):
            backend = getattr(self, f"{group}_backend")
            base_url = getattr(self, f"{group}_base_url")
            if backend == "api" and not base_url:
                missing.append(f"{group.upper()}_BASE_URL")
        if missing:
            raise ConfigError(
                "missing required *_BASE_URL for api backends: " + ", ".join(missing)
            )
