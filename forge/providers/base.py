"""Abstract provider interface for AI model calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseProvider(ABC):
    """Abstract base class for AI provider clients."""

    def __init__(self, model: str, max_tokens: int = 4096, temperature: float = 0.3) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> ProviderResponse:
        """Send messages and return a response, optionally with tool calls."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any resources (HTTP clients, connections)."""
        ...
