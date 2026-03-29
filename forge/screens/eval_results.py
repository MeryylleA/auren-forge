"""Evaluation Results screen — shows scores and allows initiating round 2."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Input, Label, Static

from forge.widgets.status_bar import ForgeStatusBar

SCORE_STARS = {5: "◆◆◆◆◆", 4: "◆◆◆◆◇", 3: "◆◆◆◇◇", 2: "◆◆◇◇◇", 1: "◆◇◇◇◇"}


def _star_rating(score: float) -> str:
    rounded = max(1, min(5, round(score)))
    return SCORE_STARS.get(rounded, "◇◇◇◇◇")


def _score_color(score: float) -> str:
    if score >= 4.0:
        return "green"
    if score >= 3.0:
        return "yellow"
    return "red"


class EvalResultsScreen(Screen):
    """Displays evaluation results and optionally kicks off a new training round."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
        Binding("a", "accept_export", "Accept & Export"),
    ]

    def __init__(self, eval_run: Any | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._eval_run = eval_run  # EvalRun model or dict

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Evaluation Results", classes="section-header")
        with VerticalScroll(id="eval-container"):
            yield Static("", id="eval-summary")
            yield Label("")
            yield Static("Scores by Dimension:", classes="bold-gold")
            yield DataTable(id="scores-table")
            yield Label("")
            yield Static("", id="agent-assessment", classes="agent-message")
            yield Label("")
            yield Static(
                "Provide a new dataset for round 2 (leave empty to accept current results):",
                classes="muted",
            )
            yield Input(
                placeholder="HuggingFace ID or local path — e.g. Open-Orca/OpenOrca",
                id="input-new-dataset",
            )
            yield Label("")
            with Horizontal():
                yield Button("🚀 Start Round 2", id="btn-round2", variant="primary")
                yield Button("✓ Accept & Export", id="btn-accept", variant="success")
                yield Button("← Back", id="btn-back", variant="default")
        yield ForgeStatusBar(
            bindings=[("Enter", "Start Round 2"), ("A", "Accept & Export"), ("Esc", "Back")]
        )

    def on_mount(self) -> None:
        table = self.query_one("#scores-table", DataTable)
        table.add_columns("Dimension", "Score", "Stars")

        # Load from arg or from session
        run = self._get_eval_run()
        if run is None:
            self.query_one("#eval-summary", Static).update(
                "[dim]No evaluation results available yet. Run [bold]forge eval[/bold] first.[/dim]"
            )
            return

        avg = run.get("avg_score", 0) if isinstance(run, dict) else run.avg_score
        passed = run.get("passed_threshold", False) if isinstance(run, dict) else run.passed_threshold
        summary = run.get("summary", "") if isinstance(run, dict) else run.summary
        scores = run.get("scores", {}) if isinstance(run, dict) else run.scores

        color = _score_color(avg)
        stars = _star_rating(avg)
        pass_tag = "[green]PASSED[/green]" if passed else "[red]BELOW THRESHOLD[/red]"
        self.query_one("#eval-summary", Static).update(
            f"Overall Score: [{color}][bold]{avg:.2f} / 5.0[/bold][/]  {stars}  {pass_tag}"
        )

        for dim, score in scores.items():
            c = _score_color(score)
            table.add_row(
                dim.replace("_", " ").title(),
                f"[{c}]{score:.2f}[/]",
                _star_rating(score),
            )

        self.query_one("#agent-assessment", Static).update(
            f"🤖 Agent Assessment:\n{summary}"
        )

    def _get_eval_run(self) -> Any | None:
        if self._eval_run:
            return self._eval_run
        sm = self.app.session_manager
        if sm and sm._state and sm.state.eval_results:
            return sm.state.eval_results[-1]
        return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-round2":
            self._start_round2()
        elif event.button.id == "btn-accept":
            self.action_accept_export()
        elif event.button.id == "btn-back":
            self.action_go_back()

    def _start_round2(self) -> None:
        new_dataset = self.query_one("#input-new-dataset", Input).value.strip()
        if not new_dataset:
            self.notify(
                "Enter a dataset source for round 2, or press 'Accept & Export' to finish.",
                severity="warning",
            )
            return

        config = self.app.config
        if config is None:
            self.notify("No config loaded.", severity="error")
            return

        # Update dataset in config for round 2
        from forge.config import DatasetConfig
        config.dataset = DatasetConfig(source=new_dataset, format="auto")
        self.app.config = config

        # Finalize previous round in session
        sm = self.app.session_manager
        if sm:
            run = self._get_eval_run()
            avg = (run.avg_score if hasattr(run, "avg_score") else run.get("avg_score")) if run else None
            sm.finalize_round(
                dataset=config.dataset.source,
                final_loss=sm.state.metrics_summary.best_loss,
                eval_score=avg,
            )

        self.notify(f"Starting round 2 with dataset: {new_dataset}", severity="information")
        from forge.screens.training import TrainingScreen
        self.app.switch_screen(TrainingScreen(config=config))

    def action_accept_export(self) -> None:
        from forge.screens.history import HistoryScreen
        self.notify("Results accepted. Use History screen to export.", severity="information")
        self.app.switch_screen(HistoryScreen())

    def action_go_back(self) -> None:
        self.app.pop_screen()
