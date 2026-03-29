"""Home / Dashboard screen — the landing page."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static

from forge.widgets.status_bar import ForgeStatusBar


BANNER = """\
   █████╗ ██╗   ██╗██████╗ ███████╗███╗   ██╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔══██╗██║   ██║██╔══██╗██╔════╝████╗  ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ███████║██║   ██║██████╔╝█████╗  ██╔██╗ ██║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██╔══██║██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ██║  ██║╚██████╔╝██║  ██║███████╗██║ ╚████║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝\
"""


class HomeScreen(Screen):
    """Main landing screen with navigation menu and recent sessions."""

    BINDINGS = [
        Binding("n", "new_project", "New Project"),
        Binding("r", "resume_session", "Resume"),
        Binding("h", "show_history", "History"),
        Binding("s", "show_settings", "Settings"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="home-container"):
            yield Static(BANNER, id="home-banner")
            yield Static(
                "AI-Orchestrated SFT Post-Training  ·  Auren Research",
                id="home-subtitle",
            )
            with Center():
                with Vertical(classes="home-menu"):
                    yield Button(
                        "⚡  New Project — Start a new SFT training run",
                        id="btn-new",
                        classes="home-menu-item primary",
                        variant="primary",
                    )
                    yield Button(
                        "↻  Resume Session — Continue from last checkpoint",
                        id="btn-resume",
                        classes="home-menu-item",
                        variant="default",
                    )
                    yield Button(
                        "📋  History — View past training rounds",
                        id="btn-history",
                        classes="home-menu-item",
                        variant="default",
                    )
                    yield Button(
                        "⚙  Settings — Configure providers & defaults",
                        id="btn-settings",
                        classes="home-menu-item",
                        variant="default",
                    )
            yield Label("")
            yield Static("Recent Sessions:", classes="section-header")
            yield DataTable(id="recent-sessions")
        yield ForgeStatusBar(
            bindings=[("N", "New"), ("R", "Resume"), ("H", "History"), ("S", "Settings"), ("Q", "Quit")]
        )

    def on_mount(self) -> None:
        table = self.query_one("#recent-sessions", DataTable)
        table.add_columns("Project", "Status", "Loss", "Round", "Updated")
        self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        table = self.query_one("#recent-sessions", DataTable)
        table.clear()
        sessions = self.app.find_recent_sessions()
        if not sessions:
            table.add_row("(no sessions found)", "", "", "", "")
            return
        for s in sessions[:10]:
            project = s.get("project_name", "unknown")
            status = s.get("status", "?")
            loss = s.get("metrics_summary", {}).get("latest_loss")
            loss_str = f"{loss:.4f}" if loss is not None else "—"
            rnd = str(s.get("current_round", 1))
            updated = s.get("updated_at", "")[:16]
            table.add_row(project, status, loss_str, rnd, updated)

    # --- Actions ---

    def action_new_project(self) -> None:
        from forge.screens.setup import SetupScreen
        self.app.push_screen(SetupScreen())

    def action_resume_session(self) -> None:
        sessions = self.app.find_recent_sessions()
        resumable = [s for s in sessions if s.get("status") in ("training", "paused", "waiting_user")]
        if not resumable:
            self.notify("No resumable sessions found.", severity="warning")
            return
        # Resume the most recent one
        output_dir = resumable[0].get("output_dir", "./output")
        self._start_training_screen(output_dir, resume=True)

    def action_show_history(self) -> None:
        from forge.screens.history import HistoryScreen
        self.app.push_screen(HistoryScreen())

    def action_show_settings(self) -> None:
        from forge.screens.settings import SettingsScreen
        self.app.push_screen(SettingsScreen())

    # --- Button handlers ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-new": self.action_new_project,
            "btn-resume": self.action_resume_session,
            "btn-history": self.action_show_history,
            "btn-settings": self.action_show_settings,
        }
        handler = actions.get(event.button.id or "")
        if handler:
            handler()

    def _start_training_screen(self, output_dir: str, resume: bool = False) -> None:
        from forge.screens.training import TrainingScreen
        self.app.push_screen(TrainingScreen(output_dir=output_dir, resume=resume))
