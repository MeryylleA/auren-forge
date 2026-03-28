"""HuggingFace TrainerCallback that feeds metrics to the Training Monitor."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Provide stubs so the module is importable without transformers installed
    TrainerCallback = object  # type: ignore[assignment,misc]
    TrainerControl = Any  # type: ignore[assignment,misc]
    TrainerState = Any  # type: ignore[assignment,misc]
    TrainingArguments = Any  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from forge.monitor import TrainingMonitor
    from forge.session import SessionManager


class ForgeMonitorCallback(TrainerCallback):  # type: ignore[misc]
    """
    Bridges HuggingFace Trainer events to the ForgeMonitor.

    On every log event the callback:
      1. Forwards step/loss/grad_norm/lr to the monitor's buffer.
      2. Checks for a pending pause/resume request from the agent.
      3. Detects training-complete and notifies the monitor.
    """

    def __init__(self, session: "SessionManager", monitor: "TrainingMonitor") -> None:
        self.session = session
        self.monitor = monitor
        self._start_time: Optional[float] = None
        self._total_steps: int = 0
        self._pause_requested: bool = False
        self._resume_requested: bool = False

    # --- Request signals (called from agent tools) ---

    def request_pause(self) -> None:
        self._pause_requested = True

    def request_resume(self) -> None:
        self._resume_requested = True

    # --- HF Trainer event hooks ---

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        self._start_time = time.monotonic()
        self._total_steps = state.max_steps
        from forge.session import TrainingProgress
        self.session.update(
            status="training",
            training_progress=TrainingProgress(
                total_steps=self._total_steps,
            ),
        )
        logger.info("Training started: total_steps=%d", self._total_steps)
        return control

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if logs is None:
            return control

        step = state.global_step
        loss = logs.get("loss") or logs.get("train_loss")
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", state.epoch)

        if loss is not None:
            self.session.append_metric(step=step, loss=float(loss), grad_norm=grad_norm, lr=lr)
            self.monitor.on_step(
                step=step,
                loss=float(loss),
                grad_norm=float(grad_norm) if grad_norm is not None else None,
                lr=float(lr) if lr is not None else None,
            )

        # Update progress in session
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        remaining: Optional[float] = None
        if self._total_steps > 0 and step > 0:
            rate = elapsed / step
            remaining = rate * (self._total_steps - step)

        from forge.session import TrainingProgress
        self.session.update(
            training_progress=TrainingProgress(
                current_step=step,
                total_steps=self._total_steps,
                current_epoch=float(epoch) if epoch is not None else 0.0,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=remaining,
            )
        )

        # Handle pause/resume signals from agent tools
        if self._pause_requested:
            self._pause_requested = False
            logger.info("Pausing training at step %d (agent request)", step)
            control.should_training_stop = True

        return control

    def on_save(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        self.session.set_checkpoint(checkpoint_dir)
        logger.info("Checkpoint saved: %s", checkpoint_dir)
        return control

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        logger.info("Training ended at step %d", state.global_step)
        self.monitor.on_training_complete()
        return control
