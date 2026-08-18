from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from typing import Any

import httpx

from .types import CompletionResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter request cannot be completed or parsed."""


class AsyncOpenRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        timeout: float = 180.0,
        max_retries: int = 5,
        site_url: str | None = None,
        app_name: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    async def __aenter__(self) -> AsyncOpenRouterClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        *,
        model: str,
        fallbacks: Sequence[str],
        messages: list[dict[str, str]],
        response_format: dict[str, object],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        reasoning_effort: str | None,
        data_collection: str = "deny",
        zdr: bool = False,
    ) -> CompletionResult:
        body: dict[str, Any] = {
            "messages": messages,
            "response_format": response_format,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "provider": {
                "allow_fallbacks": True,
                "require_parameters": True,
                "data_collection": data_collection,
                "zdr": zdr,
            },
        }
        if fallbacks:
            body["models"] = list(dict.fromkeys([model, *fallbacks]))
        else:
            body["model"] = model
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}

        response = await self._post_with_retry("/chat/completions", body)
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter returned non-JSON content") from exc

        if payload.get("error"):
            raise OpenRouterError(f"OpenRouter error: {payload['error']}")

        try:
            message = payload["choices"][0]["message"]
            content = _content_as_text(message["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("OpenRouter response did not contain assistant content") from exc

        return CompletionResult(
            content=content,
            response_id=str(payload.get("id", "")),
            requested_model=model,
            actual_model=str(payload.get("model", model)),
            provider=payload.get("provider"),
            usage=payload.get("usage") or {},
            raw_response=payload,
        )

    async def _post_with_retry(self, path: str, body: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(path, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(_backoff_seconds(attempt, None))
                continue

            if response.status_code < 400:
                return response
            if response.status_code not in RETRYABLE_STATUS_CODES or attempt == self._max_retries:
                detail = response.text[:1_000]
                raise OpenRouterError(
                    f"OpenRouter HTTP {response.status_code}: {detail or response.reason_phrase}"
                )
            retry_after = response.headers.get("Retry-After")
            await asyncio.sleep(_backoff_seconds(attempt, retry_after))

        raise OpenRouterError(f"OpenRouter request failed after retries: {last_error}")


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        if parts:
            return "".join(parts)
    raise OpenRouterError("unsupported assistant content shape")


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), 120.0))
        except ValueError:
            pass
    return min(1.5 * (2**attempt) + random.random(), 30.0)
