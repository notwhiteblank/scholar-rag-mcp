from __future__ import annotations

import gc
import importlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

from scholar_rag.core.errors import ServiceUnavailableError
from scholar_rag.models.base import DEFAULT_HTTP_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAICompatClient

_JSON_SCHEMA_NAME = "schema"
_DEFAULT_MAX_NEW_TOKENS = 2048
_LOCAL_GENERATION_KWARGS = {"temperature": 0.1, "top_p": 0.9, "do_sample": True}


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


def _parse_local_json(text: str) -> Any | None:
    if not isinstance(text, str) or not text:
        return None
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", stripped)
    if match:
        stripped = match.group(1).strip()
    try:
        return json.loads(stripped)
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


class _LocalChat:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch: Any = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        auto_model = transformers.AutoModelForCausalLM
        auto_tokenizer = transformers.AutoTokenizer
        model_path = str(Path(self._model_path).resolve())
        self._tokenizer = auto_tokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = auto_model.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        assert self._tokenizer is not None
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    def _generate(self, prompt: str) -> str:
        self._ensure_loaded()
        torch: Any = importlib.import_module("torch")
        assert self._tokenizer is not None
        assert self._model is not None
        messages = [{"role": "user", "content": prompt}]
        text = self._apply_chat_template(messages)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=_DEFAULT_MAX_NEW_TOKENS,
                **_LOCAL_GENERATION_KWARGS,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict[str, Any] | None = None,
        max_retries: int = 2,
    ) -> str | dict[str, Any]:
        if json_schema is None:
            return self._generate(prompt)
        constrained_prompt = prompt + _schema_instruction(json_schema)
        for _ in range(max_retries + 1):
            generated = self._generate(constrained_prompt)
            parsed = _parse_local_json(generated)
            if isinstance(parsed, dict):
                return parsed
        raise ServiceUnavailableError("chat output invalid json after retries")

    def health(self) -> bool:
        try:
            self._ensure_loaded()
        except Exception:
            return False
        return True

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            torch: Any = importlib.import_module("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class ChatClient:
    def __init__(
        self,
        *,
        backend: str = "api",
        base_url: str = "",
        api_key: str = "",
        model: str = "Qwen3-8B",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
    ) -> None:
        self._api: _ApiChat | None = None
        self._local: _LocalChat | None = None
        if backend == "api":
            self._api = _ApiChat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            self._local = _LocalChat(model_path=model)

    def complete(
        self,
        prompt: str,
        *,
        json_schema: dict | None = None,  # type: ignore[type-arg]
        max_retries: int = 2,
    ) -> str | dict:  # type: ignore[type-arg]
        if self._api is not None:
            return self._api.complete(
                prompt, json_schema=json_schema, max_retries=max_retries
            )
        assert self._local is not None
        return self._local.complete(
            prompt, json_schema=json_schema, max_retries=max_retries
        )

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
