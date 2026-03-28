"""Session manager: persistent state.json CRUD and crash recovery."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- State models ---


class TrainingProgress(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None


class MetricsSummary(BaseModel):
    latest_loss: Optional[float] = None
    best_loss: Optional[float] = None
    trend: str = "unknown"  # decreasing | plateau | increasing | unstable | unknown
    grad_norm_avg: Optional[float] = None
    learning_rate: Optional[float] = None
    step_count: int = 0


class DecisionEntry(BaseModel):
    timestamp: str
    trigger: str
    agent_reasoning: str
    action_taken: str


class EvalRun(BaseModel):
    timestamp: str
    round_num: int
    avg_score: float
    scores: dict[str, float] = Field(default_factory=dict)
    passed_threshold: bool = False
    summary: str = ""


class RoundHistory(BaseModel):
    round_num: int
    dataset: str
    final_loss: Optional[float] = None
    eval_score: Optional[float] = None
    completed_at: Optional[str] = None
    notes: str = ""


class SessionState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = Field(default_factory=lambda: _now())
    updated_at: str = Field(default_factory=lambda: _now())

    status: str = "configuring"
    # configuring | training | paused | evaluating | waiting_user | completed | failed

    config_path: Optional[str] = None
    project_name: str = "my-sft-project"
    model_name: str = ""
    dataset_source: str = ""
    output_dir: str = "./output"
    autonomy_level: str = "suggest"
    current_round: int = 1

    training_progress: TrainingProgress = Field(default_factory=TrainingProgress)
    metrics_summary: MetricsSummary = Field(default_factory=MetricsSummary)

    decisions_log: list[DecisionEntry] = Field(default_factory=list)
    eval_results: list[EvalRun] = Field(default_factory=list)
    round_history: list[RoundHistory] = Field(default_factory=list)

    pending_user_action: Optional[str] = None
    last_agent_call: Optional[str] = None
    last_checkpoint_path: Optional[str] = None

    loss_history: list[dict[str, Any]] = Field(default_factory=list)
    # Each entry: {"step": int, "loss": float, "grad_norm": float, "lr": float, "ts": str}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- SessionManager ---


class SessionManager:
    """Manages the persistent state.json file for a training run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / "state.json"
        self._lock = threading.Lock()
        self._state: Optional[SessionState] = None

    # --- Init / load ---

    def create_new(self, **kwargs: Any) -> SessionState:
        """Create a fresh session state."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._state = SessionState(**kwargs)
        self._write()
        logger.info("Created new session %s", self._state.session_id)
        return self._state

    def load(self) -> SessionState:
        """Load state from state.json. Raises FileNotFoundError if absent."""
        if not self.state_path.exists():
            raise FileNotFoundError(f"No state.json found at {self.state_path}")
        with open(self.state_path) as f:
            data = json.load(f)
        self._state = SessionState.model_validate(data)
        logger.info("Loaded session %s (status=%s)", self._state.session_id, self._state.status)
        return self._state

    def load_or_create(self, **kwargs: Any) -> tuple[SessionState, bool]:
        """Load existing state or create a new one. Returns (state, is_resumed)."""
        if self.state_path.exists():
            try:
                state = self.load()
                return state, True
            except Exception as e:
                logger.warning("Could not load existing state: %s — creating new session", e)
        state = self.create_new(**kwargs)
        return state, False

    def exists(self) -> bool:
        return self.state_path.exists()

    # --- Read / write ---

    @property
    def state(self) -> SessionState:
        if self._state is None:
            raise RuntimeError("Session not initialized. Call create_new() or load() first.")
        return self._state

    def update(self, **kwargs: Any) -> SessionState:
        """Update fields on the current state and persist."""
        with self._lock:
            for key, value in kwargs.items():
                if not hasattr(self._state, key):
                    raise AttributeError(f"SessionState has no field '{key}'")
                setattr(self._state, key, value)
            self._state.updated_at = _now()
            self._write()
        return self._state

    def set_status(self, status: str) -> None:
        self.update(status=status)
        logger.info("Session %s status → %s", self.state.session_id, status)

    def append_metric(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Append a metrics datapoint to loss_history (kept to last 1000 points)."""
        with self._lock:
            entry: dict[str, Any] = {"step": step, "loss": loss, "ts": _now()}
            if grad_norm is not None:
                entry["grad_norm"] = grad_norm
            if lr is not None:
                entry["lr"] = lr
            self._state.loss_history.append(entry)
            # Keep memory bounded
            if len(self._state.loss_history) > 1000:
                self._state.loss_history = self._state.loss_history[-1000:]
            self._state.updated_at = _now()
            self._write()

    def log_decision(self, trigger: str, reasoning: str, action: str) -> None:
        entry = DecisionEntry(
            timestamp=_now(),
            trigger=trigger,
            agent_reasoning=reasoning,
            action_taken=action,
        )
        with self._lock:
            self._state.decisions_log.append(entry)
            self._state.last_agent_call = _now()
            self._state.updated_at = _now()
            self._write()

    def add_eval_result(self, result: EvalRun) -> None:
        with self._lock:
            self._state.eval_results.append(result)
            self._state.updated_at = _now()
            self._write()

    def finalize_round(self, dataset: str, final_loss: Optional[float], eval_score: Optional[float]) -> None:
        """Record a completed training round before starting a new one."""
        with self._lock:
            entry = RoundHistory(
                round_num=self._state.current_round,
                dataset=dataset,
                final_loss=final_loss,
                eval_score=eval_score,
                completed_at=_now(),
            )
            self._state.round_history.append(entry)
            self._state.current_round += 1
            self._state.loss_history = []  # reset for new round
            self._state.updated_at = _now()
            self._write()

    def update_metrics_summary(self, summary: MetricsSummary) -> None:
        with self._lock:
            self._state.metrics_summary = summary
            self._state.updated_at = _now()
            self._write()

    def set_pending_user_action(self, description: Optional[str]) -> None:
        self.update(pending_user_action=description)

    def set_checkpoint(self, path: str) -> None:
        self.update(last_checkpoint_path=path)

    def get_compact_state(self) -> dict[str, Any]:
        """Return a compact representation for injecting into agent prompts."""
        s = self.state
        return {
            "session_id": s.session_id,
            "status": s.status,
            "project": s.project_name,
            "model": s.model_name,
            "dataset": s.dataset_source,
            "autonomy": s.autonomy_level,
            "current_round": s.current_round,
            "progress": {
                "step": s.training_progress.current_step,
                "total": s.training_progress.total_steps,
                "epoch": round(s.training_progress.current_epoch, 3),
                "elapsed_min": round(s.training_progress.elapsed_seconds / 60, 1),
            },
            "metrics": {
                "latest_loss": s.metrics_summary.latest_loss,
                "best_loss": s.metrics_summary.best_loss,
                "trend": s.metrics_summary.trend,
                "grad_norm_avg": s.metrics_summary.grad_norm_avg,
                "lr": s.metrics_summary.learning_rate,
            },
            "last_decisions": [
                {
                    "ts": d.timestamp,
                    "trigger": d.trigger,
                    "action": d.action_taken,
                }
                for d in s.decisions_log[-5:]
            ],
            "eval_results": [
                {
                    "round": e.round_num,
                    "avg_score": e.avg_score,
                    "passed": e.passed_threshold,
                }
                for e in s.eval_results
            ],
            "pending_user_action": s.pending_user_action,
            "last_checkpoint": s.last_checkpoint_path,
        }

    def get_recent_loss_history(self, last_n: int = 100) -> list[dict[str, Any]]:
        return self.state.loss_history[-last_n:]

    # --- Internal ---

    def _write(self) -> None:
        """Write state to disk atomically using a temp file."""
        tmp = self.state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._state.model_dump(), f, indent=2)
        tmp.replace(self.state_path)
