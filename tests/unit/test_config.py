import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from scholar_rag.core.config import Settings
from scholar_rag.core.errors import ConfigError


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SCHOLAR_RAG_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))


def write_config(data_dir: Path, payload: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.json").write_text(json.dumps(payload))


def _home(*parts: str) -> Path:
    return Path.home().joinpath(*parts)


DEFAULTS: list[tuple[str, Callable[[], object]]] = [
    ("data_dir", lambda: _home(".scholar-rag")),
    ("qdrant_url", lambda: ""),
    ("qdrant_bin", lambda: ""),
    ("qdrant_port", lambda: 6333),
    ("qdrant_storage_dir", lambda: _home(".local", "share", "scholar-rag", "qdrant")),
    ("chat_backend", lambda: "api"),
    ("chat_base_url", lambda: ""),
    ("chat_api_key", lambda: ""),
    ("chat_model", lambda: "Qwen3.5-0.8B"),
    ("embed_backend", lambda: "api"),
    ("embed_base_url", lambda: ""),
    ("embed_api_key", lambda: ""),
    ("embed_model", lambda: "jina-embeddings-v5-text-small"),
    ("rerank_backend", lambda: "api"),
    ("rerank_base_url", lambda: ""),
    ("rerank_api_key", lambda: ""),
    ("rerank_model", lambda: "jina-reranker-v3.5"),
    ("mineru_backend", lambda: "python"),
    ("mineru_api_url", lambda: "http://127.0.0.1:8010"),
    ("mineru_bin", lambda: "mineru"),
    ("keywords_enabled", lambda: True),
    ("annotation_resolver_enabled", lambda: True),
    ("job_workers", lambda: 1),
    ("crossref_enabled", lambda: True),
    ("grobid_url", lambda: ""),
    ("log_level", lambda: "WARNING"),
    ("log_dir", lambda: ""),
]


@pytest.mark.parametrize(
    "field,expected_factory",
    DEFAULTS,
    ids=[name for name, _ in DEFAULTS],
)
def test_default_value(field, expected_factory):
    settings = Settings()
    assert getattr(settings, field) == expected_factory()


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_MODEL", "env-model")
    monkeypatch.setenv("SCHOLAR_RAG_QDRANT_PORT", "7000")
    monkeypatch.setenv("SCHOLAR_RAG_KEYWORDS_ENABLED", "false")
    settings = Settings()
    assert settings.chat_model == "env-model"
    assert settings.qdrant_port == 7000
    assert settings.keywords_enabled is False


def test_config_json_is_low_priority_source():
    write_config(Path.home() / ".scholar-rag", {"CHAT_MODEL": "file-model", "job_workers": 3})
    settings = Settings.load()
    assert settings.chat_model == "file-model"
    assert settings.job_workers == 3


def test_env_vars_override_config_json(monkeypatch):
    write_config(Path.home() / ".scholar-rag", {"CHAT_MODEL": "file-model"})
    monkeypatch.setenv("SCHOLAR_RAG_CHAT_MODEL", "env-model")
    settings = Settings.load()
    assert settings.chat_model == "env-model"


def test_data_dir_priority_env_then_file_then_default(monkeypatch):
    env_dir = Path.home() / "env-dir"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "config.json").write_text(json.dumps({"CHAT_MODEL": "file-in-env-dir"}))
    monkeypatch.setenv("SCHOLAR_RAG_DATA_DIR", str(env_dir))
    settings = Settings.load()
    assert settings.data_dir == env_dir
    assert settings.chat_model == "file-in-env-dir"


def test_missing_config_json_falls_back_to_defaults():
    settings = Settings.load()
    assert settings.data_dir == Path.home() / ".scholar-rag"
    assert settings.chat_model == "Qwen3.5-0.8B"


def test_validate_runtime_lists_all_missing_base_urls():
    settings = Settings()
    with pytest.raises(ConfigError) as exc_info:
        settings.validate_runtime()
    message = str(exc_info.value)
    assert "CHAT_BASE_URL" in message
    assert "EMBED_BASE_URL" in message
    assert "RERANK_BASE_URL" in message


def test_validate_runtime_passes_when_base_urls_set():
    settings = Settings(
        chat_base_url="http://chat",
        embed_base_url="http://embed",
        rerank_base_url="http://rerank",
    )
    settings.validate_runtime()


def test_validate_runtime_passes_when_backends_are_local():
    settings = Settings(chat_backend="local", embed_backend="local", rerank_backend="local")
    settings.validate_runtime()


@pytest.mark.parametrize("env_var,value", [
    ("SCHOLAR_RAG_CHAT_BACKEND", "magic"),
    ("SCHOLAR_RAG_EMBED_BACKEND", "magic"),
    ("SCHOLAR_RAG_RERANK_BACKEND", "magic"),
    ("SCHOLAR_RAG_MINERU_BACKEND", "magic"),
])
def test_invalid_backend_env_raises_config_error(monkeypatch, env_var, value):
    monkeypatch.setenv(env_var, value)
    with pytest.raises(ConfigError):
        Settings()


def test_invalid_backend_in_config_json_raises_config_error():
    write_config(Path.home() / ".scholar-rag", {"RERANK_BACKEND": "bogus"})
    with pytest.raises(ConfigError):
        Settings.load()
