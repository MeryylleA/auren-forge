"""Tests for forge.monitor module."""

import math
from unittest.mock import MagicMock, patch

import pytest

from forge.monitor import (
    AnomalyType,
    TrainingMonitor,
    _compute_trend,
    _linear_slope,
)
from forge.session import SessionManager


@pytest.fixture
def session(tmp_path):
    mgr = SessionManager(tmp_path / "output")
    mgr.create_new(project_name="test", model_name="llama", dataset_source="test/ds")
    return mgr


@pytest.fixture
def monitor(session):
    return TrainingMonitor(session=session, check_interval_minutes=0)


# --- _linear_slope ---


def test_linear_slope_increasing():
    values = [float(i) for i in range(10)]
    slope = _linear_slope(values)
    assert slope > 0


def test_linear_slope_flat():
    values = [1.0] * 10
    slope = _linear_slope(values)
    assert abs(slope) < 1e-9


def test_linear_slope_decreasing():
    values = [float(10 - i) for i in range(10)]
    slope = _linear_slope(values)
    assert slope < 0


def test_linear_slope_single_point():
    assert _linear_slope([5.0]) == 0.0


# --- _compute_trend ---


def test_compute_trend_decreasing():
    values = [10.0 - i * 0.01 for i in range(300)]
    trend = _compute_trend(values)
    assert trend == "decreasing"


def test_compute_trend_plateau():
    values = [2.0 + i * 1e-9 for i in range(300)]
    trend = _compute_trend(values)
    assert trend == "plateau"


def test_compute_trend_few_points():
    assert _compute_trend([1.0, 2.0]) == "unknown"


# --- Anomaly detection ---


def test_nan_loss_triggers_agent(session):
    triggered = []

    def fake_trigger(event, ctx=None):
        triggered.append(event)

    monitor = TrainingMonitor(
        session=session,
        agent_trigger_callback=fake_trigger,
        check_interval_minutes=0,
    )
    monitor.on_step(step=1, loss=float("nan"))
    assert AnomalyType.NAN_LOSS in triggered


def test_inf_loss_triggers_agent(session):
    triggered = []

    def fake_trigger(event, ctx=None):
        triggered.append(event)

    monitor = TrainingMonitor(
        session=session,
        agent_trigger_callback=fake_trigger,
        check_interval_minutes=0,
    )
    monitor.on_step(step=1, loss=float("inf"))
    assert AnomalyType.INF_LOSS in triggered


def test_loss_spike_detection(session):
    """Loss spike: current_loss > 3× rolling avg."""
    triggered = []

    def fake_trigger(event, ctx=None):
        triggered.append(event)

    monitor = TrainingMonitor(
        session=session,
        agent_trigger_callback=fake_trigger,
        check_interval_minutes=0,
    )
    # Feed 60 stable steps with loss ~1.0
    for i in range(60):
        monitor.on_step(step=i, loss=1.0)

    # Now trigger a spike — reset debounce to well before the 300s window
    triggered.clear()
    import time as _time
    monitor._last_agent_wake = _time.monotonic() - 400
    monitor.on_step(step=61, loss=10.0)
    assert AnomalyType.LOSS_SPIKE in triggered


def test_training_complete_triggers_agent(session):
    triggered = []

    def fake_trigger(event, ctx=None):
        triggered.append(event)

    monitor = TrainingMonitor(
        session=session,
        agent_trigger_callback=fake_trigger,
        check_interval_minutes=0,
    )
    import time as _time
    monitor._last_agent_wake = _time.monotonic() - 400  # ensure debounce window has passed
    monitor.on_training_complete()
    assert "training_complete" in triggered


def test_no_duplicate_wake_debounce(session):
    """Non-critical events should be debounced within 5 minutes."""
    triggered = []

    import time
    def fake_trigger(event, ctx=None):
        triggered.append(event)

    monitor = TrainingMonitor(
        session=session,
        agent_trigger_callback=fake_trigger,
        check_interval_minutes=0,
    )
    monitor._last_agent_wake = time.monotonic()  # simulate recent wake

    # Feed 60 stable then spike — should be debounced
    for i in range(60):
        monitor.on_step(step=i, loss=1.0)
    monitor.on_step(step=61, loss=10.0)

    # Spike was detected but debounced
    spike_events = [e for e in triggered if e == AnomalyType.LOSS_SPIKE]
    assert len(spike_events) == 0


def test_metrics_summary_updated(session):
    monitor = TrainingMonitor(
        session=session,
        check_interval_minutes=0,
    )
    for i in range(110):
        monitor.on_step(step=i, loss=2.0 - i * 0.001)

    summary = session.state.metrics_summary
    assert summary.latest_loss is not None
    assert summary.best_loss is not None
    assert summary.best_loss <= summary.latest_loss
