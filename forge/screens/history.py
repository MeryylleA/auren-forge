"""History screen — browse and manage past training sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Label, Static

from forge.widgets.status_bar import ForgeStatusBar


class HistoryScreen(Screen):
    """Browse past training sessions. Select one to view details or export."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("x", "export_selected", "Export"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sessions: list[dict[str, Any]] = []
        self._selected_row: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Training History", classes="section-header")
        with Vertical():
            yield DataTable(id="history-table", cursor_type="row")
            yield Label("")
            yield Static("", id="detail-panel", classes="muted")
            yield Label("")
            with Horizontal():
                yield Button("↻ Refresh", id="btn-refresh", variant="default")
                yield Button("📤 Export Selected", id="btn-export", variant="default")
                yield Button("▶ Resume Selected", id="btn-resume", variant="primary")
        yield ForgeStatusBar(
            bindings=[("↑↓", "Navigate"), ("R", "Refresh"), ("X", "Export"), ("Esc", "Back")]
        )

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Project", "Status", "Rounds", "Best Loss", "Eval Score", "Updated")
        self._load_sessions()

    def _load_sessions(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.clear()
        self._sessions = self.app.find_recent_sessions()

        if not self._sessions:
            table.add_row("(no sessions found)", "", "", "", "", "")
            return

        for s in self._sessions:
            project = s.get("project_name", "unknown")
            status = s.get("status", "?")
            rounds = str(s.get("current_round", 1))
            metrics = s.get("metrics_summary", {})
            best_loss = metrics.get("best_loss")
            best_str = f"{best_loss:.4f}" if best_loss is not None else "—"

            evals = s.get("eval_results", [])
            if evals:
                last_score = evals[-1].get("avg_score") if isinstance(evals[-1], dict) else evals[-1].avg_score
                eval_str = f"{last_score:.2f}" if last_score else "—"
            else:
                eval_str = "—"

            updated = s.get("updated_at", "")[:16]

            status_markup = {
                "training": "[green]running[/]",
                "completed": "[bright_green]completed[/]",
                "paused": "[yellow]paused[/]",
                "failed": "[red]failed[/]",
                "evaluating": "[blue]evaluating[/]",
                "waiting_user": "[magenta]waiting[/]",
            }.get(status, status)

            table.add_row(project, status_markup, rounds, best_str, eval_str, updated)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._selected_row = event.cursor_row
        self._show_detail(event.cursor_row)

    def _show_detail(self, row: int) -> None:
        detail = self.query_one("#detail-panel", Static)
        if row >= len(self._sessions):
            detail.update("")
            return

        s = self._sessions[row]
        model = s.get("model_name", "unknown")
        dataset = s.get("dataset_source", "unknown")
        session_id = s.get("session_id", "?")
        output_dir = s.get("output_dir", "./output")
        checkpoint = s.get("last_checkpoint_path", "none")

        detail.update(
            f"[bold gold1]Session:[/] {session_id}  "
            f"[bold gold1]Model:[/] {model}  "
            f"[bold gold1]Dataset:[/] {dataset}\n"
            f"[bold gold1]Output:[/] {output_dir}  "
            f"[bold gold1]Checkpoint:[/] {checkpoint}"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-refresh":
            self.action_refresh()
        elif event.button.id == "btn-export":
            self.action_export_selected()
        elif event.button.id == "btn-resume":
            self._resume_selected()

    def action_refresh(self) -> None:
        self._load_sessions()
        self.notify("Session list refreshed.", severity="information")

    def action_export_selected(self) -> None:
        if self._selected_row >= len(self._sessions):
            self.notify("No session selected.", severity="warning")
            return
        s = self._sessions[self._selected_row]
        output_dir = s.get("output_dir", "./output")
        self._do_export(output_dir)

    def _do_export(self, output_dir: str) -> None:
        config = self.app.config
        if config is None:
            self.notify("No config loaded — cannot export.", severity="error")
            return

        def _export_worker() -> str:
            from forge.session import SessionManager
            from forge.training import ForgeTrainer
            sm = SessionManager(output_dir)
            sm.load()
            trainer = ForgeTrainer(config=config, session=sm)
            trainer.load_model()
            path = trainer.export(fmt="safetensors")
            return path

        self.notify("Exporting model (safetensors)…", severity="information")
        worker = self.run_worker(_export_worker, thread=True)

    def _resume_selected(self) -> None:
        if self._selected_row >= len(self._sessions):
            self.notify("No session selected.", severity="warning")
            return
        s = self._sessions[self._selected_row]
        output_dir = s.get("output_dir", "./output")

        config = self.app.config
        if config is None:
            from forge.config import find_config, load_config
            path = find_config()
            if path:
                config = load_config(path)
                self.app.config = config
            else:
                self.notify("No config found — cannot resume.", severity="error")
                return

        from forge.screens.training import TrainingScreen
        self.app.push_screen(TrainingScreen(output_dir=output_dir, resume=True, config=config))

    def action_go_back(self) -> None:
        self.app.pop_screen()
