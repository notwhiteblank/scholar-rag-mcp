from __future__ import annotations

import logging
import sys

from scholar_rag.core.config import Settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

_settings: Settings | None = None


def _load_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def _resolve_level(value: str) -> int:
    level = logging.getLevelName(value.upper())
    if isinstance(level, int):
        return level
    return logging.WARNING


def get_logger(name: str) -> logging.Logger:
    settings = _load_settings()
    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    root.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    return logger
