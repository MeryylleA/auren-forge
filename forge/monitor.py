"""Training monitor: metrics parsing, anomaly detection, and agent wake-up triggers."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable, Optional

from forge.session import MetricsSummary, SessionManager

if TYPE_CHECKING:
    from forge.agent import AgentOrchestrator

logger = logging.getLogger(__name__)

# --- Anomaly thresholds ---
SPIKE_MULTIPLIER = 3.0       # loss > 3× rolling avg → spike
PLATEAU_SLOPE_THRESHOLD = 1e-6  # |slope| < this over last 500 steps → plateau
PLATEAU_MIN_STEPS = 500
DIVERGENCE_MIN_STEPS = 500   # increasing loss for 500+ consecutive steps
GRAD_NORM_SPIKE_MULTIPLIER = 10.0
ROLLING_WINDOW = 50          # steps for rolling loss avg
METRICS_UPDATE_INTERVAL = 100  # update session every N steps


class AnomalyType:
    NAN_LOSS = "nan_loss"
    INF_LOSS = "inf_loss"
    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_DIVERGENCE = "loss_divergence"
    GRAD_NORM_SPIKE = "grad_norm_spike"


class TrainingMonitor:
    """
    Lightweight process (no LLM calls) that:
    - Buffers metrics
    - Detects anomalies
    - Updates state.json periodically
    - Wakes the agent on relevant events
    """

    def __init__(
        self,
        session: SessionManager,
        agent_trigger_callback: Optional[Callable[[str, Optional[dict[str, Any]]], None]] = None,
        check_interval_minutes: int = 30,
    ) -> None:
        self.session = session
        self._trigger_callback = agent_trigger_callback
        self._check_interval_sec = check_interval_minutes * 60

        self._loss_buffer: deque[float] = deque(maxlen=1000)
        self._grad_norm_buffer: deque[float] = deque(maxlen=200)
        self._step_buffer: deque[int] = deque(maxlen=1000)

        self._step_counter: int = 0
        self._best_loss: Optional[float] = None
        self._last_agent_wake: float = 0.0
        self._consecutive_increasing: int = 0
        self._training_complete: bool = False
        self._lock = threading.Lock()

        # Periodic scheduled-interval timer
        self._schedule_timer: Optional[threading.Timer] = None
        if self._check_interval_sec > 0:
            self._arm_timer()

    # --- Public API ---

    def on_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        lr: Optional[float] = None,
    ) -> None:
        """Called after every training step log event."""
        with self._lock:
            self._step_counter += 1
            self._step_buffer.append(step)
            self._loss_buffer.append(loss)
            if grad_norm is not None:
                self._grad_norm_buffer.append(grad_norm)

            # Check NaN / Inf first (critical)
            if math.isnan(loss):
                self._wake_agent(AnomalyType.NAN_LOSS, {"step": step, "loss": loss})
                return
            if math.isinf(loss):
                self._wake_agent(AnomalyType.INF_LOSS, {"step": step, "loss": loss})
                return

            # Update best loss
            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss

            anomaly = self._check_anomalies(step, loss, grad_norm)
            if anomaly:
                self._wake_agent(anomaly, {"step": step, "loss": loss, "grad_norm": grad_norm})

            # Periodic metrics summary update
            if self._step_counter % METRICS_UPDATE_INTERVAL == 0:
                self._flush_metrics_summary(lr)

    def on_training_complete(self) -> None:
        """Called when training finishes normally."""
        with self._lock:
            self._training_complete = True
            self._cancel_timer()
        self._flush_metrics_summary()
        self._wake_agent("training_complete", {"message": "Training run finished"})

    def attach_agent(self, agent: "AgentOrchestrator") -> None:
        """Attach an agent orchestrator to receive wake-up triggers."""
        def trigger(event: str, ctx: Optional[dict[str, Any]] = None) -> None:
            try:
                agent.call_sync(trigger=event, extra_context=ctx)
            except Exception as e:
                logger.error("Agent call failed for trigger '%s': %s", event, e)

        self._trigger_callback = trigger

    def notify_user_message(self, message: str) -> None:
        """Called when the user sends a message during training."""
        self._wake_agent("user_message", {"user_message": message})

    def shutdown(self) -> None:
        self._cancel_timer()

    # --- Internal ---

    def _check_anomalies(
        self, step: int, loss: float, grad_norm: Optional[float]
    ) -> Optional[str]:
        buf = list(self._loss_buffer)

        # Spike detection
        if len(buf) >= ROLLING_WINDOW:
            rolling_avg = sum(buf[-ROLLING_WINDOW:-1]) / (ROLLING_WINDOW - 1)
            if rolling_avg > 0 and loss > SPIKE_MULTIPLIER * rolling_avg:
                logger.warning(
                    "Loss spike at step %d: %.4f vs rolling avg %.4f", step, loss, rolling_avg
                )
                return AnomalyType.LOSS_SPIKE

        # Divergence: N consecutive increasing steps
        if len(buf) >= 2 and buf[-1] > buf[-2]:
            self._consecutive_increasing += 1
        else:
            self._consecutive_increasing = 0

        if self._consecutive_increasing >= DIVERGENCE_MIN_STEPS:
            logger.warning("Loss divergence detected at step %d", step)
            return AnomalyType.LOSS_DIVERGENCE

        # Plateau detection (requires enough data)
        if len(buf) >= PLATEAU_MIN_STEPS:
            slope = _linear_slope(buf[-PLATEAU_MIN_STEPS:])
            if abs(slope) < PLATEAU_SLOPE_THRESHOLD:
                logger.info("Loss plateau detected at step %d (slope=%.2e)", step, slope)
                return AnomalyType.LOSS_PLATEAU

        # Gradient norm spike
        gnorm_buf = list(self._grad_norm_buffer)
        if grad_norm is not None and len(gnorm_buf) >= ROLLING_WINDOW:
            avg_gnorm = sum(gnorm_buf[-ROLLING_WINDOW:-1]) / (ROLLING_WINDOW - 1)
            if avg_gnorm > 0 and grad_norm > GRAD_NORM_SPIKE_MULTIPLIER * avg_gnorm:
                logger.warning(
                    "Grad norm spike at step %d: %.2f vs avg %.2f", step, grad_norm, avg_gnorm
                )
                return AnomalyType.GRAD_NORM_SPIKE

        return None

    def _flush_metrics_summary(self, lr: Optional[float] = None) -> None:
        buf = list(self._loss_buffer)
        if not buf:
            return

        latest = buf[-1]
        avg_recent = sum(buf[-ROLLING_WINDOW:]) / min(len(buf), ROLLING_WINDOW)
        trend = _compute_trend(buf)

        gnorm_buf = list(self._grad_norm_buffer)
        avg_gnorm = sum(gnorm_buf) / len(gnorm_buf) if gnorm_buf else None

        summary = MetricsSummary(
            latest_loss=round(latest, 6),
            best_loss=round(self._best_loss, 6) if self._best_loss else None,
            trend=trend,
            grad_norm_avg=round(avg_gnorm, 4) if avg_gnorm is not None else None,
            learning_rate=lr,
            step_count=len(buf),
        )
        self.session.update_metrics_summary(summary)

    def _wake_agent(
        self, trigger: str, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Wake the agent if one is attached (debounce: skip if called too recently for non-critical)."""
        now = time.monotonic()
        critical_triggers = {AnomalyType.NAN_LOSS, AnomalyType.INF_LOSS}

        if trigger not in critical_triggers:
            # Debounce non-critical triggers: at least 5 minutes between wakes
            if now - self._last_agent_wake < 300:
                logger.debug("Debouncing agent wake for trigger '%s'", trigger)
                return

        self._last_agent_wake = now
        logger.info("Waking agent: trigger=%s", trigger)
        if self._trigger_callback:
            try:
                self._trigger_callback(trigger, context)
            except Exception as e:
                logger.error("Agent trigger callback failed: %s", e)

    def _arm_timer(self) -> None:
        self._schedule_timer = threading.Timer(
            self._check_interval_sec, self._scheduled_check
        )
        self._schedule_timer.daemon = True
        self._schedule_timer.start()

    def _cancel_timer(self) -> None:
        if self._schedule_timer is not None:
            self._schedule_timer.cancel()
            self._schedule_timer = None

    def _scheduled_check(self) -> None:
        if not self._training_complete:
            self._flush_metrics_summary()
            self._wake_agent("scheduled_interval", {"message": "Periodic status check"})
            self._arm_timer()


# --- Math helpers ---


def _linear_slope(values: list[float]) -> float:
    """Compute the slope of a simple linear regression over the values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_trend(buf: list[float]) -> str:
    """Classify the trend of recent losses."""
    if len(buf) < 10:
        return "unknown"
    recent = buf[-min(200, len(buf)):]
    slope = _linear_slope(recent)

    if abs(slope) < PLATEAU_SLOPE_THRESHOLD:
        return "plateau"
    if slope < -1e-5:
        return "decreasing"
    if slope > 1e-5:
        return "increasing"
    return "unstable"
