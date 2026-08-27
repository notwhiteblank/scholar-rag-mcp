from __future__ import annotations

import logging
import sys
from pathlib import Path

from scholar_rag.core.config import Settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_FILE_LOG_NAME = "scholar-rag.log"

_settings: Settings | None = None
_file_log_configured = False


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


def _configure_file_log(settings: Settings) -> None:
    global _file_log_configured
    if _file_log_configured or not settings.log_dir:
        return
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / _FILE_LOG_NAME, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    _file_log_configured = True


def get_logger(name: str) -> logging.Logger:
    settings = _load_settings()
    level = _resolve_level(settings.log_level)
    root = logging.getLogger()
    _configure_file_log(settings)
    effective = logging.DEBUG if settings.log_dir else level
    root.setLevel(effective)
    logger = logging.getLogger(name)
    logger.setLevel(effective)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    return logger
