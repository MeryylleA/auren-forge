"""Ollama Cloud API client (OpenAI-compatible) for auren-forge."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

from forge.providers.base import BaseProvider, ProviderResponse, ToolCall

logger = logging.getLogger(__name__)

OLLAMA_CLOUD_BASE_URL = "https://api.ollama.com/v1"


class OllamaProvider(BaseProvider):
    """Provider that calls models via Ollama Cloud's OpenAI-compatible API."""

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_CLOUD_BASE_URL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        self._base_url = base_url.rstrip("/")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )
        # Track whether the model supports native tool calling
        self._supports_tools: Optional[bool] = None

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

        # Attempt native tool calling; fall back to prompt-based if unsupported
        if tools:
            if self._supports_tools is not False:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

        logger.debug("Ollama request: model=%s, messages=%d", self.model, len(messages))

        try:
            resp = await self._client.post("/chat/completions", content=json.dumps(payload))
        except httpx.RequestError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        if resp.status_code == 400 and tools and self._supports_tools is not False:
            # Model doesn't support native tool calling — retry with prompt injection
            logger.info("Model %s doesn't support native tools; using prompt-based fallback", self.model)
            self._supports_tools = False
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            if tools:
                payload["messages"] = _inject_tools_into_prompt(messages, tools)
            resp = await self._client.post("/chat/completions", content=json.dumps(payload))

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            body = e.response.text
            logger.error("Ollama HTTP error %s: %s", e.response.status_code, body)
            raise RuntimeError(f"Ollama API error {e.response.status_code}: {body}") from e

        if tools and self._supports_tools is None:
            self._supports_tools = True

        data = resp.json()
        return _parse_response(data, prompt_based=self._supports_tools is False)

    async def close(self) -> None:
        await self._client.aclose()


def _parse_response(data: dict[str, Any], prompt_based: bool = False) -> ProviderResponse:
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
            ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args)
        )

    if prompt_based and content and not tool_calls:
        tool_calls = _extract_prompt_based_tool_calls(content)

    usage = data.get("usage", {})
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )


def _inject_tools_into_prompt(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Inject tool definitions into the system message for models without native tool support."""
    tool_desc = json.dumps(tools, indent=2)
    injection = (
        "\n\n## Available Tools\nYou can call these tools by outputting JSON in this exact format:\n"
        '```json\n{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}\n```\n'
        f"Tool definitions:\n{tool_desc}"
    )
    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = {**result[0], "content": result[0]["content"] + injection}
    else:
        result.insert(0, {"role": "system", "content": injection})
    return result


def _extract_prompt_based_tool_calls(content: str) -> list[ToolCall]:
    """Try to parse prompt-injected tool calls from model output."""
    import re

    pattern = re.compile(r'```json\s*(\{.*?"tool_call".*?\})\s*```', re.DOTALL)
    calls: list[ToolCall] = []
    for match in pattern.finditer(content):
        try:
            obj = json.loads(match.group(1))
            tc = obj.get("tool_call", {})
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if name:
                calls.append(ToolCall(id="prompt-0", name=name, arguments=args))
        except (json.JSONDecodeError, AttributeError):
            pass
    return calls
