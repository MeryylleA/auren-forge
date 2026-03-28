"""Tests for forge.agent module."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.agent import AgentOrchestrator, _format_args, _short_result, build_provider
from forge.config import AgentConfig
from forge.providers.base import ProviderResponse, ToolCall
from forge.session import SessionManager
from forge.tools import ToolExecutor


@pytest.fixture
def session(tmp_path):
    mgr = SessionManager(tmp_path / "output")
    mgr.create_new(
        project_name="agent-test",
        model_name="llama",
        dataset_source="test/ds",
        autonomy_level="suggest",
    )
    return mgr


@pytest.fixture
def executor(session):
    return ToolExecutor(session=session)


# --- _format_args ---


def test_format_args_empty():
    assert _format_args({}) == ""


def test_format_args_simple():
    result = _format_args({"a": 1, "b": "x"})
    assert "a=1" in result
    assert "b='x'" in result


# --- _short_result ---


def test_short_result_ok():
    assert _short_result({"ok": True}) == "ok"


def test_short_result_error():
    r = _short_result({"error": "something broke"})
    assert "ERROR" in r


def test_short_result_data():
    r = _short_result({"count": 5, "status": "done"})
    assert "count" in r


# --- AgentOrchestrator ---


@pytest.mark.asyncio
async def test_agent_no_tool_calls(session, executor):
    """Agent returns immediately when LLM responds with no tool calls."""
    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(
        return_value=ProviderResponse(
            content="Training is looking good. No action needed.",
            tool_calls=[],
            finish_reason="stop",
        )
    )

    agent = AgentOrchestrator(
        provider=mock_provider, session=session, executor=executor
    )
    result = await agent.call(trigger="scheduled_interval")

    assert "No actions taken" in result
    assert len(session.state.decisions_log) == 1
    assert session.state.decisions_log[0].trigger == "scheduled_interval"


@pytest.mark.asyncio
async def test_agent_with_tool_call(session, executor):
    """Agent executes a tool call and logs the decision."""
    get_status_response = ProviderResponse(
        content="Let me check status.",
        tool_calls=[ToolCall(id="tc1", name="get_training_status", arguments={})],
        finish_reason="tool_calls",
    )
    final_response = ProviderResponse(
        content="Status looks fine.",
        tool_calls=[],
        finish_reason="stop",
    )

    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(side_effect=[get_status_response, final_response])

    agent = AgentOrchestrator(
        provider=mock_provider, session=session, executor=executor
    )
    result = await agent.call(trigger="anomaly_check")

    assert "get_training_status" in result
    assert len(session.state.decisions_log) == 1


@pytest.mark.asyncio
async def test_agent_max_tool_rounds(session, executor):
    """Agent stops after MAX_TOOL_ROUNDS even if it keeps calling tools."""
    from forge.agent import MAX_TOOL_ROUNDS

    # Always return a tool call
    tool_response = ProviderResponse(
        content=None,
        tool_calls=[ToolCall(id="tc", name="get_training_status", arguments={})],
        finish_reason="tool_calls",
    )

    mock_provider = MagicMock()
    mock_provider.chat = AsyncMock(return_value=tool_response)

    agent = AgentOrchestrator(
        provider=mock_provider, session=session, executor=executor
    )
    result = await agent.call(trigger="loop_test")

    # Should complete without infinite loop
    assert mock_provider.chat.call_count <= MAX_TOOL_ROUNDS


# --- ToolExecutor ---


def test_tool_get_training_status(session, executor):
    result = executor.execute("get_training_status", {})
    assert "status" in result
    assert "step" in result


def test_tool_get_loss_history_empty(session, executor):
    result = executor.execute("get_loss_history", {"last_n_steps": 10})
    assert result["count"] == 0
    assert result["history"] == []


def test_tool_send_notification(session, executor):
    result = executor.execute(
        "send_notification",
        {"message": "Test notification", "priority": "info"},
    )
    assert result["ok"] is True
    notifications = executor.pop_notifications()
    assert len(notifications) == 1
    assert notifications[0]["message"] == "Test notification"


def test_tool_send_notification_requires_response(session, executor):
    executor.execute(
        "send_notification",
        {"message": "Need input", "requires_response": True, "priority": "warning"},
    )
    assert session.state.pending_user_action == "Need input"


def test_tool_get_session_summary(session, executor):
    result = executor.execute("get_session_summary", {})
    assert "session_id" in result
    assert "status" in result


def test_tool_unknown(session, executor):
    result = executor.execute("nonexistent_tool", {})
    assert "error" in result


def test_pop_notifications_clears_queue(session, executor):
    executor.execute("send_notification", {"message": "msg1"})
    executor.execute("send_notification", {"message": "msg2"})
    first = executor.pop_notifications()
    assert len(first) == 2
    second = executor.pop_notifications()
    assert len(second) == 0


# --- build_provider ---


def test_build_provider_openrouter():
    config = AgentConfig(
        provider="openrouter",
        api_key="sk-test",
        model="anthropic/claude-opus-4.6",
    )
    provider = build_provider(config)
    from forge.providers.openrouter import OpenRouterProvider
    assert isinstance(provider, OpenRouterProvider)


def test_build_provider_openrouter_no_key():
    config = AgentConfig(provider="openrouter", api_key=None, model="some/model")
    with pytest.raises(ValueError, match="api_key"):
        build_provider(config)


def test_build_provider_ollama():
    config = AgentConfig(provider="ollama", model="kimi-k2.5:cloud")
    provider = build_provider(config)
    from forge.providers.ollama import OllamaProvider
    assert isinstance(provider, OllamaProvider)
