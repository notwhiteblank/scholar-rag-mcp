from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from qdrant_client import models

from scholar_rag.core.errors import InvalidFilterError

_WHITELIST = ("doc_id", "section", "doi", "journal", "title", "first_author", "year", "added_after")
_KEYWORD_FIELDS = ("doc_id", "section", "doi")
_NORM_FIELDS = ("journal", "title", "first_author")


def _whitelist_message() -> str:
    return ", ".join(_WHITELIST)


def _keyword_condition(field: str, value: Any) -> models.FieldCondition:
    if isinstance(value, str):
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise InvalidFilterError(f"filter field {field!r} expects string values")
        return models.FieldCondition(key=field, match=models.MatchAny(any=value))
    raise InvalidFilterError(f"filter field {field!r} expects a string or a list of strings")


def _norm_condition(field: str, value: Any) -> models.FieldCondition:
    key = f"{field}_norm"
    if isinstance(value, str):
        return models.FieldCondition(key=key, match=models.MatchValue(value=value.lower()))
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise InvalidFilterError(f"filter field {field!r} expects string values")
        lowered = [item.lower() for item in value]
        return models.FieldCondition(key=key, match=models.MatchAny(any=lowered))
    raise InvalidFilterError(f"filter field {field!r} expects a string or a list of strings")


def _year_condition(value: Any) -> models.FieldCondition:
    if isinstance(value, list):
        raise InvalidFilterError("filter field 'year' does not accept lists")
    if isinstance(value, int) and not isinstance(value, bool):
        return models.FieldCondition(key="year", match=models.MatchValue(value=value))
    if isinstance(value, dict):
        if not set(value).issubset({"gte", "lte"}):
            raise InvalidFilterError("filter field 'year' range accepts only 'gte' and 'lte'")
        bounds: dict[str, int] = {}
        for key in ("gte", "lte"):
            bound = value.get(key)
            if bound is None:
                continue
            if not isinstance(bound, int) or isinstance(bound, bool):
                raise InvalidFilterError(f"filter field 'year' range bound {key!r} must be an integer")
            bounds[key] = bound
        return models.FieldCondition(key="year", range=models.Range(**bounds))
    raise InvalidFilterError("filter field 'year' expects an integer or a range dict with 'gte'/'lte'")


def _added_after_condition(value: Any) -> models.FieldCondition:
    if not isinstance(value, str):
        raise InvalidFilterError("filter field 'added_after' expects an ISO8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidFilterError(f"filter field 'added_after' is not a valid ISO8601 date: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return models.FieldCondition(key="added_ts", range=models.Range(gte=parsed.timestamp()))


def translate_filter(f: dict) -> Any:  # type: ignore[type-arg]
    if not f:
        return None
    conditions: list[Any] = []
    for field, value in f.items():
        if field in _KEYWORD_FIELDS:
            conditions.append(_keyword_condition(field, value))
        elif field in _NORM_FIELDS:
            conditions.append(_norm_condition(field, value))
        elif field == "year":
            conditions.append(_year_condition(value))
        elif field == "added_after":
            conditions.append(_added_after_condition(value))
        else:
            raise InvalidFilterError(f"unknown filter field {field!r}; allowed fields: {_whitelist_message()}")
    return models.Filter(must=conditions)
