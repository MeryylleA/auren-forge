"""Scrollable agent decisions log widget."""

from __future__ import annotations

from typing import Any

from textual.widgets import RichLog

from rich.text import Text


class AgentLog(RichLog):
    """Scrollable log of agent decisions and notifications."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, highlight=True, markup=True, wrap=True, **kwargs)

    def add_decision(self, timestamp: str, trigger: str, action: str) -> None:
        """Add a decision entry to the log."""
        ts = timestamp[:19] if len(timestamp) > 19 else timestamp
        self.write(f"[dim]{ts}[/dim]  [cyan]{trigger}[/cyan]  {action}")

    def add_notification(self, message: str, priority: str = "info") -> None:
        """Add an agent notification to the log."""
        icons = {"info": "[blue]ℹ[/blue]", "warning": "[yellow]⚠[/yellow]", "critical": "[red]✗[/red]"}
        icon = icons.get(priority, icons["info"])
        self.write(f"{icon}  {message}")

    def add_agent_message(self, message: str) -> None:
        """Add an agent chat response."""
        self.write(f"[#87ceeb]🤖 Forge:[/]  {message}")

    def add_system_event(self, message: str) -> None:
        """Add a system event (training start, stop, etc)."""
        self.write(f"[dim]>>> {message}[/dim]")

    def load_decisions(self, decisions: list[dict[str, Any]]) -> None:
        """Load a batch of existing decisions into the log."""
        for d in decisions:
            ts = d.get("timestamp", d.get("ts", ""))
            trigger = d.get("trigger", "")
            action = d.get("action_taken", d.get("action", ""))
            self.add_decision(ts, trigger, action)
