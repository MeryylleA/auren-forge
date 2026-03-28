"""Tests for forge.session module."""

import json
import time
from pathlib import Path

import pytest

from forge.session import (
    DecisionEntry,
    EvalRun,
    MetricsSummary,
    RoundHistory,
    SessionManager,
    SessionState,
    TrainingProgress,
)


@pytest.fixture
def mgr(tmp_path):
    return SessionManager(tmp_path / "output")


# --- create_new ---


def test_create_new_writes_state_json(mgr, tmp_path):
    state = mgr.create_new(project_name="test", model_name="llama")
    assert (mgr.output_dir / "state.json").exists()
    assert state.project_name == "test"
    assert state.model_name == "llama"
    assert state.status == "configuring"


def test_create_new_generates_session_id(mgr):
    state = mgr.create_new()
    assert len(state.session_id) == 8


# --- load ---


def test_load_roundtrip(mgr):
    mgr.create_new(project_name="load-test")
    mgr2 = SessionManager(mgr.output_dir)
    state = mgr2.load()
    assert state.project_name == "load-test"


def test_load_missing_raises(tmp_path):
    mgr = SessionManager(tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError):
        mgr.load()


# --- update ---


def test_update_status(mgr):
    mgr.create_new()
    mgr.set_status("training")
    assert mgr.state.status == "training"
    # Persisted
    loaded = SessionManager(mgr.output_dir)
    loaded.load()
    assert loaded.state.status == "training"


def test_update_unknown_field_raises(mgr):
    mgr.create_new()
    with pytest.raises(AttributeError):
        mgr.update(nonexistent_field="oops")


# --- append_metric ---


def test_append_metric(mgr):
    mgr.create_new()
    mgr.append_metric(step=10, loss=2.5, grad_norm=1.2, lr=0.0002)
    assert len(mgr.state.loss_history) == 1
    assert mgr.state.loss_history[0]["step"] == 10
    assert mgr.state.loss_history[0]["loss"] == 2.5


def test_append_metric_bounded(mgr):
    mgr.create_new()
    for i in range(1100):
        mgr.append_metric(step=i, loss=float(i))
    assert len(mgr.state.loss_history) <= 1000


# --- log_decision ---


def test_log_decision(mgr):
    mgr.create_new()
    mgr.log_decision(trigger="anomaly", reasoning="loss spike", action="paused")
    assert len(mgr.state.decisions_log) == 1
    d = mgr.state.decisions_log[0]
    assert d.trigger == "anomaly"
    assert d.action_taken == "paused"


# --- add_eval_result ---


def test_add_eval_result(mgr):
    mgr.create_new()
    run = EvalRun(
        timestamp="2024-01-01T00:00:00+00:00",
        round_num=1,
        avg_score=3.7,
        passed_threshold=True,
        summary="Good",
    )
    mgr.add_eval_result(run)
    assert len(mgr.state.eval_results) == 1
    assert mgr.state.eval_results[0].avg_score == 3.7


# --- finalize_round ---


def test_finalize_round(mgr):
    mgr.create_new(current_round=1)
    mgr.finalize_round(dataset="test/ds", final_loss=0.5, eval_score=3.8)
    assert len(mgr.state.round_history) == 1
    assert mgr.state.current_round == 2
    assert mgr.state.round_history[0].dataset == "test/ds"


# --- get_compact_state ---


def test_get_compact_state_keys(mgr):
    mgr.create_new(project_name="compact-test", model_name="llama")
    compact = mgr.get_compact_state()
    for key in ("session_id", "status", "project", "model", "progress", "metrics"):
        assert key in compact


# --- atomic write (tmp file) ---


def test_atomic_write_uses_tmp(mgr, tmp_path, monkeypatch):
    """Verify no .tmp file lingers after a write."""
    mgr.create_new()
    tmp_file = mgr.state_path.with_suffix(".json.tmp")
    assert not tmp_file.exists()
