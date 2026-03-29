"""Agent Chat screen — interactive conversation with the Forge agent."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Header, Input, Label, Static
from textual.worker import Worker, WorkerState

from forge.widgets.agent_log import AgentLog
from forge.widgets.status_bar import ForgeStatusBar

logger = logging.getLogger(__name__)


class ChatScreen(Screen):
    """Interactive chat with the Forge agent about the current session."""

    BINDINGS = [
        Binding("escape", "close_chat", "Back"),
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Chat with Forge Agent", classes="section-header")
        with Vertical():
            yield AgentLog(id="chat-log")
            yield Label(" Message:", classes="bold-gold")
            yield Input(
                placeholder="Ask anything about the training session...",
                id="chat-input",
            )
        yield ForgeStatusBar(
            bindings=[("Enter", "Send"), ("Esc", "Back"), ("Ctrl+L", "Clear")]
        )

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", AgentLog)

        # Greet with session context if available
        sm = self.app.session_manager
        if sm and sm._state:
            state = sm.state
            metrics = state.metrics_summary
            progress = state.training_progress
            loss_str = f"{metrics.latest_loss:.4f}" if metrics.latest_loss else "N/A"
            step_str = f"{progress.current_step:,}/{progress.total_steps:,}"
            greeting = (
                f"Hello! I'm monitoring your '{state.project_name}' training run. "
                f"Current status: {state.status}. "
                f"Step {step_str}, loss {loss_str}, trend: {metrics.trend}. "
                "Ask me anything — I can explain decisions, analyze metrics, or take actions."
            )
        else:
            greeting = (
                "Hello! I'm Forge, your AI training assistant. "
                "Start a training run to get session-aware insights. "
                "You can ask me general questions about SFT training in the meantime."
            )

        log.add_agent_message(greeting)
        self.query_one("#chat-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        event.input.clear()
        self._send_message(message)

    def _send_message(self, message: str) -> None:
        log = self.query_one("#chat-log", AgentLog)
        ts = datetime.now(timezone.utc).strftime("%H:%M")
        log.write(f"[bold white]You ({ts}):[/bold white]  {message}")
        log.add_system_event("Forge is thinking...")
        self.run_worker(
            lambda: self._agent_call(message),
            thread=True,
            group="chat",
        )

    def _agent_call(self, user_message: str) -> str:
        """Runs in background thread."""
        agent = self.app.agent
        if agent is None:
            # No live session — try to create a minimal agent from app config
            config = self.app.config
            if config is None:
                return "No active session or config found. Start a training run first."
            from forge.agent import AgentOrchestrator, build_provider
            from forge.session import SessionManager
            from forge.tools import ToolExecutor
            sm = self.app.session_manager
            if sm is None:
                sm = SessionManager(config.training.output_dir)
                if sm.exists():
                    sm.load()
                else:
                    return "No session state found. Start a training run first."
            provider = build_provider(config.agent)
            executor = ToolExecutor(session=sm)
            agent = AgentOrchestrator(provider=provider, session=sm, executor=executor)

        result = agent.call_sync(
            trigger="user_chat",
            extra_context={"user_message": user_message},
        )

        # Pull any tool-triggered notifications
        if self.app.executor:
            notifications = self.app.executor.pop_notifications()
            for n in notifications:
                self.app.call_from_thread(self._show_notification, n["message"])

        return result

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "chat":
            return
        log = self.query_one("#chat-log", AgentLog)

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result or "(no response)"
            # Remove the "thinking..." placeholder and add real response
            self.app.call_from_thread(self._replace_thinking, result)

        elif event.state == WorkerState.ERROR:
            self.app.call_from_thread(
                self._replace_thinking, f"Error: {event.worker.error}"
            )

    def _replace_thinking(self, response: str) -> None:
        log = self.query_one("#chat-log", AgentLog)
        # Write the real response (the thinking message is already appended)
        log.add_agent_message(response)

    def _show_notification(self, message: str) -> None:
        log = self.query_one("#chat-log", AgentLog)
        log.add_notification(message)

    def action_close_chat(self) -> None:
        self.app.pop_screen()

    def action_clear_log(self) -> None:
        self.query_one("#chat-log", AgentLog).clear()
        self.on_mount()
