"""OpenRouter API client (OpenAI-compatible) for auren-forge."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from forge.providers.base import BaseProvider, ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """Provider that calls models via OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/meryyllea/auren-forge",
                "X-Title": "auren-forge",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        logger.debug("OpenRouter request: model=%s, messages=%d", self.model, len(messages))

        try:
            resp = await self._client.post("/chat/completions", content=json.dumps(payload))
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text
            logger.error("OpenRouter HTTP error %s: %s", e.response.status_code, body)
            raise RuntimeError(f"OpenRouter API error {e.response.status_code}: {body}") from e
        except httpx.RequestError as e:
            logger.error("OpenRouter request error: %s", e)
            raise RuntimeError(f"OpenRouter request failed: {e}") from e

        data = resp.json()
        return _parse_response(data)

    async def close(self) -> None:
        await self._client.aclose()


def _parse_response(data: dict[str, Any]) -> ProviderResponse:
    choice = data["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")

    content = message.get("content")
    tool_calls: list[ToolCall] = []

    raw_tool_calls = message.get("tool_calls") or []
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {"raw": raw_args}
        tool_calls.append(
            ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            )
        )

    usage = data.get("usage", {})
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )
