from __future__ import annotations

import time
from typing import Any

import httpx

from scholar_rag.core.errors import ConfigError, ServiceUnavailableError

DEFAULT_TIMEOUT = 120.0
DEFAULT_HTTP_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        if not base_url:
            raise ConfigError("base_url is required for the api backend")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        response: httpx.Response | None = None
        last_error: httpx.TransportError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(path, headers=self._headers(), json=body)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            if response.status_code < 500:
                return response
            if attempt < self.max_retries:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
        if last_error is not None:
            raise ServiceUnavailableError(
                f"connection to {self.base_url} failed after {self.max_retries} retries: {last_error}"
            ) from last_error
        if response is None:
            raise ServiceUnavailableError("request failed without a response")
        raise ServiceUnavailableError(
            f"service returned HTTP {response.status_code} after {self.max_retries} retries"
        )

    def health(self) -> bool:
        try:
            response = self._client.get("/models", headers=self._headers())
        except httpx.HTTPError:
            return False
        return 200 <= response.status_code < 300

    def close(self) -> None:
        self._client.close()
