"""Live Training Monitor screen — the main workhorse screen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Static
from textual.worker import Worker, WorkerState

from forge.widgets.agent_log import AgentLog
from forge.widgets.loss_chart import LossChart
from forge.widgets.metrics_panel import MetricsPanel
from forge.widgets.status_bar import ForgeStatusBar

logger = logging.getLogger(__name__)


class TrainingScreen(Screen):
    """Live training monitor with metrics, loss chart, and agent decisions."""

    BINDINGS = [
        Binding("p", "toggle_pause", "Pause/Resume"),
        Binding("e", "run_eval", "Eval"),
        Binding("c", "open_chat", "Chat"),
        Binding("escape", "go_home", "Home"),
    ]

    class TrainingFinished(Message):
        pass

    class MetricsUpdated(Message):
        def __init__(self, state: dict[str, Any]) -> None:
            super().__init__()
            self.state = state

    def __init__(
        self,
        output_dir: str = "./output",
        resume: bool = False,
        config: Any = None,
        config_path: str = "forge.yaml",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._output_dir = output_dir
        self._resume = resume
        self._config = config
        self._config_path = config_path
        self._training_worker: Optional[Worker] = None
        self._is_paused = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="training-title", classes="section-header")
        with Horizontal():
            with Vertical(id="left-col"):
                yield MetricsPanel(id="metrics-panel")
                yield Static(" Loss Curve:", classes="bold-gold")
                yield LossChart(id="loss-chart")
            with Vertical(id="right-col"):
                yield Static(" Agent Decisions:", classes="bold-gold")
                yield AgentLog(id="agent-log")
        yield ForgeStatusBar(
            bindings=[("P", "Pause"), ("E", "Eval"), ("C", "Chat"), ("Esc", "Home")],
        )

    def on_mount(self) -> None:
        title = self.query_one("#training-title", Static)

        if self._config is None:
            from forge.config import load_config, find_config
            path = find_config() or self._config_path
            try:
                self._config = load_config(path)
            except Exception as e:
                self.notify(f"Cannot load config: {e}", severity="error")
                return
            self._output_dir = self._config.training.output_dir

        project = self._config.project_name if self._config else "unknown"
        title.update(f" AUREN FORGE — Training: {project}")

        agent_log = self.query_one("#agent-log", AgentLog)
        agent_log.add_system_event("Initializing training session...")

        # Start periodic metrics refresh
        self.set_interval(2.0, self._refresh_metrics)

        # Launch training in a background worker
        self._training_worker = self.run_worker(
            self._run_training, thread=True, group="training"
        )

    def _run_training(self) -> str:
        """Runs in a background thread — sets up and executes the full training pipeline."""
        config = self._config
        output_dir = Path(config.training.output_dir)

        from forge.session import SessionManager
        session_mgr = SessionManager(output_dir)

        if self._resume and session_mgr.exists():
            state, _ = session_mgr.load_or_create()
        else:
            state = session_mgr.create_new(
                project_name=config.project_name,
                model_name=config.model.name,
                dataset_source=config.dataset.source,
                output_dir=str(output_dir),
                autonomy_level=config.agent.autonomy,
                config_path=self._config_path,
            )

        # Store on app for other screens to use
        self.app.session_manager = session_mgr
        self.app.config = config

        from forge.agent import AgentOrchestrator, build_provider
        from forge.callback import ForgeMonitorCallback
        from forge.dataset import DatasetHandler
        from forge.evaluation import Evaluator
        from forge.monitor import TrainingMonitor
        from forge.tools import ToolExecutor
        from forge.training import ForgeTrainer

        trainer = ForgeTrainer(config=config, session=session_mgr)
        dataset_handler = DatasetHandler(
            model_name=config.model.name,
            column_mapping=config.dataset.column_mapping or {},
        )
        provider = build_provider(config.agent)
        executor = ToolExecutor(session=session_mgr)
        executor.set_trainer(trainer)
        executor.set_dataset_handler(dataset_handler)

        monitor = TrainingMonitor(
            session=session_mgr,
            check_interval_minutes=config.agent.check_interval_minutes,
        )
        agent = AgentOrchestrator(provider=provider, session=session_mgr, executor=executor)
        monitor.attach_agent(agent)

        self.app.trainer = trainer
        self.app.provider = provider
        self.app.agent = agent
        self.app.executor = executor
        self.app.monitor = monitor

        self.app.call_from_thread(self._log_system, "Loading model and applying LoRA...")
        trainer.load_model()
        self.app.call_from_thread(self._log_system, f"Model loaded: {config.model.name}")

        evaluator = Evaluator(
            provider=provider,
            trainer=trainer,
            session=session_mgr,
            min_quality_score=config.eval.min_quality_score,
        )
        executor.set_evaluator(evaluator)

        self.app.call_from_thread(self._log_system, f"Validating dataset: {config.dataset.source}")
        validation = dataset_handler.validate(
            source=config.dataset.source,
            fmt=config.dataset.format,
            split=config.dataset.split,
        )
        if not validation["valid"]:
            issues = "; ".join(validation["issues"][:3])
            self.app.call_from_thread(
                self._log_agent, f"Dataset issues: {issues}"
            )
        else:
            self.app.call_from_thread(
                self._log_system,
                f"Dataset OK: {validation['num_rows']:,} rows, format={validation['format']}",
            )

        self.app.call_from_thread(self._log_system, "Formatting dataset with chat template...")
        effective_fmt = config.dataset.format if config.dataset.format != "auto" else validation["format"]
        prepared_ds = dataset_handler.prepare_for_training(
            tokenizer=trainer.tokenizer,
            fmt=effective_fmt,
            max_seq_length=config.model.max_seq_length,
        )
        self.app.call_from_thread(
            self._log_system, f"Dataset prepared: {len(prepared_ds):,} examples"
        )

        callback = ForgeMonitorCallback(session=session_mgr, monitor=monitor)
        trainer.build_trainer(dataset=prepared_ds, callback=callback)

        agent.call_sync("training_start", {"config": config.model_dump(mode="json")})

        self.app.call_from_thread(self._log_system, "Training started!")

        if self._resume and state.last_checkpoint_path:
            self.app.call_from_thread(
                self._log_system, f"Resuming from: {state.last_checkpoint_path}"
            )
            trainer.resume_from_checkpoint(state.last_checkpoint_path)
        else:
            trainer.train()

        self.app.call_from_thread(self._log_system, "Training loop complete.")

        # Post-training eval
        if config.agent.eval_on_completion:
            session_mgr.set_status("evaluating")
            self.app.call_from_thread(self._log_system, "Running post-training evaluation...")
            eval_result = evaluator.run(num_prompts=config.eval.num_test_prompts)
            self.app.call_from_thread(
                self._log_agent,
                f"Eval score: {eval_result.avg_score:.2f}/5.0 — {eval_result.summary}",
            )
            if not eval_result.passed_threshold:
                session_mgr.set_pending_user_action(
                    f"Quality {eval_result.avg_score:.2f} below threshold. Provide new dataset?"
                )

        session_mgr.set_status("completed")
        monitor.shutdown()

        self.app.call_from_thread(self._log_system, "Session complete.")
        return "done"

    # --- Thread-safe logging helpers (called via app.call_from_thread) ---

    def _log_system(self, msg: str) -> None:
        try:
            log = self.query_one("#agent-log", AgentLog)
            log.add_system_event(msg)
        except Exception:
            pass

    def _log_agent(self, msg: str) -> None:
        try:
            log = self.query_one("#agent-log", AgentLog)
            log.add_agent_message(msg)
        except Exception:
            pass

    # --- Periodic metrics refresh ---

    def _refresh_metrics(self) -> None:
        """Poll session state and update widgets."""
        sm = self.app.session_manager
        if sm is None or sm._state is None:
            return

        state = sm.state
        progress = state.training_progress
        metrics = state.metrics_summary

        # Update metrics panel
        panel = self.query_one("#metrics-panel", MetricsPanel)
        panel.update_metrics(
            step=progress.current_step,
            total_steps=progress.total_steps,
            epoch=progress.current_epoch,
            loss=metrics.latest_loss,
            best_loss=metrics.best_loss,
            trend=metrics.trend,
            lr=metrics.learning_rate,
            grad_norm=metrics.grad_norm_avg,
            elapsed_min=progress.elapsed_seconds / 60,
            eta_min=(
                progress.estimated_remaining_seconds / 60
                if progress.estimated_remaining_seconds is not None
                else None
            ),
            round_num=state.current_round,
            status=state.status,
        )

        # Update loss chart
        chart = self.query_one("#loss-chart", LossChart)
        losses = [pt["loss"] for pt in state.loss_history if "loss" in pt]
        if losses:
            chart.update_values(losses)

        # Update status bar for pause state
        bar = self.query_one(ForgeStatusBar)
        pause_label = "Resume" if state.status == "paused" else "Pause"
        bar.set_bindings(
            [("P", pause_label), ("E", "Eval"), ("C", "Chat"), ("Esc", "Home")]
        )

        # Update title
        title = self.query_one("#training-title", Static)
        status_tag = state.status.upper()
        project = state.project_name
        title.update(f" AUREN FORGE — Training: {project} — {status_tag}")

        # Load new agent decisions
        log = self.query_one("#agent-log", AgentLog)
        for notif in (self.app.executor.pop_notifications() if self.app.executor else []):
            log.add_notification(notif["message"], notif.get("priority", "info"))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.ERROR:
            self._log_system(f"Training error: {event.worker.error}")
            self.notify(f"Training failed: {event.worker.error}", severity="error")

    # --- Actions ---

    def action_toggle_pause(self) -> None:
        trainer = self.app.trainer
        sm = self.app.session_manager
        if trainer is None or sm is None:
            return
        if sm.state.status == "paused":
            trainer.request_resume()
            sm.set_status("training")
            self._log_system("Training resumed by user.")
            self._is_paused = False
        else:
            trainer.request_pause()
            sm.set_status("paused")
            self._log_system("Training paused by user.")
            self._is_paused = True

    def action_run_eval(self) -> None:
        self.notify("Triggering manual evaluation...", severity="information")
        self._log_system("Manual evaluation triggered.")
        if self.app.agent:
            self.run_worker(self._call_agent_eval, thread=True)

    def _call_agent_eval(self) -> None:
        if self.app.agent:
            self.app.agent.call_sync("user_requested_eval")

    def action_open_chat(self) -> None:
        from forge.screens.chat import ChatScreen
        self.app.push_screen(ChatScreen())

    def action_go_home(self) -> None:
        self.app.pop_screen()
