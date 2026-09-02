from __future__ import annotations

import json
from typing import Any

import httpx

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

_JSON_SCHEMA_NAME = "schema"


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "\n\nOutput a single JSON object conforming to the following JSON schema:\n"
        + json.dumps(schema)
    )


def _extract_content(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, TypeError, IndexError):
        return None
    if isinstance(content, str):
        return content
    return None


def _parse_json(content: str) -> Any | None:
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def _status_error_message(operation: str, status_code: int, retries: int) -> str:
    return f"{operation} failed: HTTP {status_code} after {retries} retries"


class _ApiChat(OpenAICompatClient):
    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_retries: int = 2,
    ) -> str | dict[str, Any]:
        if json_schema is None:
            return self._complete_plain(prompt)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": _JSON_SCHEMA_NAME,
                    "schema": json_schema,
                    "strict": True,
                },
            },
        }
        response = self._post("/chat/completions", body)
        if response.status_code < 400:
            content = _extract_content(response)
            if content is not None:
                parsed = _parse_json(content)
                if isinstance(parsed, dict):
                    return parsed
        constrained_prompt = prompt + _schema_instruction(json_schema)
        for _ in range(max_retries + 1):
            retry_response = self._post(
                "/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": constrained_prompt}],
                },
            )
            if retry_response.status_code >= 400:
                raise ServiceUnavailableError(
                    _status_error_message(
                        "chat completion", retry_response.status_code, max_retries
                    )
                )
            content = _extract_content(retry_response)
            if content is not None:
                parsed = _parse_json(content)
                if isinstance(parsed, dict):
                    return parsed
        raise ServiceUnavailableError("chat output invalid json after retries")

    def _complete_plain(self, prompt: str) -> str:
        response = self._post(
            "/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": prompt}]},
        )
        if response.status_code >= 400:
            raise ServiceUnavailableError(
                _status_error_message(
                    "chat completion", response.status_code, self.max_retries
                )
            )
        content = _extract_content(response)
        if content is None:
            raise ServiceUnavailableError("chat completion returned no content")
        return content


class ChatClient:
    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        model: str = "Qwen3.5-0.8B",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        self._api = _ApiChat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,  # type: ignore[type-arg]
        max_retries: int = 2,
    ) -> str | dict:  # type: ignore[type-arg]
        return self._api.complete(
            prompt, json_schema=json_schema, max_retries=max_retries
        )

    def health(self) -> bool:
        return self._api.health()

    def close(self) -> None:
        self._api.close()
