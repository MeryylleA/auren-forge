"""AI Agent orchestrator: stateless LLM calls with tool dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from forge.providers.base import BaseProvider, ProviderResponse
from forge.tools import TOOL_SCHEMAS, ToolExecutor

if TYPE_CHECKING:
    from forge.session import SessionManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are Forge, an AI training engineer assistant built by Auren Research. \
You orchestrate supervised fine-tuning (SFT) of language models.

You are called periodically during training runs to monitor progress, diagnose \
issues, and make decisions. You do NOT maintain memory between calls — all context \
is provided to you in each message via the session state.

Your autonomy level for this session is: {autonomy_level}
- "monitor": You observe and report only. Never take actions without user confirmation.
- "suggest": You analyze and recommend actions, then wait for user approval before executing.
- "auto": You execute decisions within guardrails (pause on anomaly, run eval on completion, \
notify user on issues). You NEVER start a new training round without user approval.

CRITICAL RULES:
1. Base ALL decisions on the metrics and state provided. Never guess or assume.
2. A new SFT round with a different dataset ALWAYS requires explicit user approval, \
regardless of autonomy level.
3. If you detect loss divergence (NaN, Inf, or loss increasing for 500+ steps), \
pause training immediately.
4. Keep your reasoning concise. You are called frequently — don't waste tokens.
5. When evaluating model outputs, be specific about what's wrong and what type of \
data might help.
6. Log every decision with clear reasoning in the decisions_log.

Available tools are provided in the function definitions. Use them to interact with \
the training system.\
"""

MAX_TOOL_ROUNDS = 10  # prevent infinite tool-calling loops


class AgentOrchestrator:
    """Manages stateless LLM calls, tool dispatch, and decision logging."""

    def __init__(
        self,
        provider: BaseProvider,
        session: "SessionManager",
        executor: ToolExecutor,
    ) -> None:
        self.provider = provider
        self.session = session
        self.executor = executor

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            autonomy_level=self.session.state.autonomy_level
        )

    def _build_user_message(
        self,
        trigger: str,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> str:
        state = self.session.get_compact_state()
        recent_losses = self.session.get_recent_loss_history(50)
        context: dict[str, Any] = {
            "trigger": trigger,
            "session_state": state,
            "recent_loss_history": recent_losses,
        }
        if extra_context:
            context.update(extra_context)
        return json.dumps(context, indent=2)

    async def call(
        self,
        trigger: str,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Make a stateless agent call for the given trigger event.
        Handles multi-round tool calling and logs the final decision.
        Returns a summary of the actions taken.
        """
        logger.info("Agent woken: trigger=%s", trigger)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_message(trigger, extra_context)},
        ]

        actions_taken: list[str] = []
        reasoning_buffer: list[str] = []

        for round_num in range(MAX_TOOL_ROUNDS):
            response: ProviderResponse = await self.provider.chat(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )

            # Collect any text reasoning
            if response.content:
                reasoning_buffer.append(response.content)

            if not response.has_tool_calls:
                # Agent is done
                break

            # Append assistant message with tool_calls
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.content:
                assistant_msg["content"] = response.content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in response.tool_calls
            ]
            messages.append(assistant_msg)

            # Execute each tool call and append results
            for tc in response.tool_calls:
                logger.info("Agent calling tool: %s(%s)", tc.name, tc.arguments)
                result = self.executor.execute(tc.name, tc.arguments)
                actions_taken.append(f"{tc.name}({_format_args(tc.arguments)}) → {_short_result(result)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

            if round_num == MAX_TOOL_ROUNDS - 1:
                logger.warning("Agent hit max tool rounds (%d)", MAX_TOOL_ROUNDS)
                break

        # Log the decision
        reasoning = " | ".join(reasoning_buffer) if reasoning_buffer else "No text reasoning provided"
        action_summary = "; ".join(actions_taken) if actions_taken else "No actions taken"

        self.session.log_decision(
            trigger=trigger,
            reasoning=reasoning[:1000],  # cap length
            action=action_summary[:500],
        )

        logger.info("Agent call complete. Actions: %s", action_summary)
        return action_summary

    def call_sync(
        self,
        trigger: str,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Synchronous wrapper around call() for use from training callbacks."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g., Jupyter)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.call(trigger, extra_context))
                    return future.result()
            else:
                return loop.run_until_complete(self.call(trigger, extra_context))
        except RuntimeError:
            return asyncio.run(self.call(trigger, extra_context))


def build_provider(config_agent: Any) -> BaseProvider:
    """Instantiate the correct provider from AgentConfig."""
    from forge.providers.ollama import OllamaProvider
    from forge.providers.openrouter import OpenRouterProvider

    if config_agent.provider == "openrouter":
        if not config_agent.api_key:
            raise ValueError(
                "OpenRouter provider requires api_key. "
                "Set it in your config or OPENROUTER_API_KEY env var."
            )
        return OpenRouterProvider(
            api_key=config_agent.api_key,
            model=config_agent.model,
            max_tokens=config_agent.max_tokens_per_call,
            temperature=config_agent.temperature,
        )
    elif config_agent.provider == "ollama":
        base_url = config_agent.base_url or "https://api.ollama.com/v1"
        return OllamaProvider(
            model=config_agent.model,
            base_url=base_url,
            api_key=config_agent.api_key,
            max_tokens=config_agent.max_tokens_per_call,
            temperature=config_agent.temperature,
        )
    else:
        raise ValueError(f"Unknown provider: {config_agent.provider}")


def _format_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = [f"{k}={repr(v)}" for k, v in list(args.items())[:3]]
    return ", ".join(parts)


def _short_result(result: dict[str, Any]) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    if "ok" in result:
        return "ok"
    keys = list(result.keys())[:2]
    return "{" + ", ".join(f"{k}: {result[k]!r}" for k in keys) + "}"
