"""Tests for the TUI app, widgets, and entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --- Entry point ---


def test_main_import():
    """forge.__main__.main should be importable and callable as a function."""
    from forge.__main__ import main
    assert callable(main)


def test_forge_app_importable():
    from forge.app import ForgeApp
    assert ForgeApp is not None


# --- ForgeApp helpers ---


def test_load_settings_missing(tmp_path, monkeypatch):
    """load_settings returns empty dict when file doesn't exist."""
    monkeypatch.setattr("forge.app.SETTINGS_PATH", tmp_path / "nonexistent.yaml")
    from forge.app import ForgeApp
    app = ForgeApp()
    result = app.load_settings()
    assert result == {}


def test_save_and_load_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("forge.app.SETTINGS_DIR", tmp_path)
    monkeypatch.setattr("forge.app.SETTINGS_PATH", tmp_path / "settings.yaml")
    from forge.app import ForgeApp
    app = ForgeApp()
    app.save_settings({"default_model": "foo/bar", "default_autonomy": "auto"})
    result = app.load_settings()
    assert result["default_model"] == "foo/bar"
    assert result["default_autonomy"] == "auto"


def test_find_recent_sessions_empty(tmp_path):
    from forge.app import ForgeApp
    app = ForgeApp()
    result = app.find_recent_sessions(search_dirs=[str(tmp_path)])
    assert result == []


def test_find_recent_sessions_finds_state_json(tmp_path):
    from forge.app import ForgeApp
    # Create a fake state.json
    state = {
        "session_id": "abc123",
        "project_name": "test-project",
        "status": "completed",
        "updated_at": "2024-01-01T12:00:00+00:00",
        "metrics_summary": {"best_loss": 0.5},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    app = ForgeApp()
    result = app.find_recent_sessions(search_dirs=[str(tmp_path)])
    assert len(result) == 1
    assert result[0]["project_name"] == "test-project"


def test_find_recent_sessions_sorted(tmp_path):
    """Sessions should be sorted by updated_at descending."""
    from forge.app import ForgeApp
    for i, ts in enumerate(["2024-01-01T10:00:00+00:00", "2024-01-01T12:00:00+00:00"]):
        sub = tmp_path / f"run{i}"
        sub.mkdir()
        state = {"session_id": f"s{i}", "project_name": f"proj{i}", "status": "completed", "updated_at": ts}
        (sub / "state.json").write_text(json.dumps(state))
    app = ForgeApp()
    result = app.find_recent_sessions(search_dirs=[str(tmp_path)])
    assert result[0]["updated_at"] > result[1]["updated_at"]


# --- LossChart widget ---


def test_loss_chart_sparkline_empty():
    from forge.widgets.loss_chart import LossChart
    chart = LossChart()
    rendered = chart.render()
    assert "No loss data" in str(rendered)


def test_loss_chart_sparkline_values():
    from forge.widgets.loss_chart import LossChart
    chart = LossChart(chart_width=20, chart_height=4)
    chart.loss_values = [2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.85]
    rendered = str(chart.render())
    # Should contain axis labels
    assert "│" in rendered


def test_loss_chart_append_value():
    from forge.widgets.loss_chart import LossChart
    chart = LossChart()
    for v in [1.0, 0.9, 0.8]:
        chart.append_value(v)
    assert list(chart.loss_values)[-1] == pytest.approx(0.8)


def test_loss_chart_bounded():
    from forge.widgets.loss_chart import LossChart
    chart = LossChart()
    for i in range(2500):
        chart.append_value(float(i))
    assert len(chart.loss_values) <= 2000


# --- MetricsPanel widget ---


def test_metrics_panel_renders_without_data():
    from forge.widgets.metrics_panel import MetricsPanel
    panel = MetricsPanel()
    rendered = panel.render()
    assert rendered is not None


def test_metrics_panel_renders_with_data():
    from forge.widgets.metrics_panel import MetricsPanel
    panel = MetricsPanel()
    panel.update_metrics(
        step=100, total_steps=1000, epoch=0.1,
        loss=1.23, best_loss=1.10, trend="decreasing",
        lr=2e-4, grad_norm=0.5, elapsed_min=5.0, eta_min=50.0,
        round_num=1, status="training",
    )
    # Render should not raise
    table = panel.render()
    assert table is not None


# --- AgentLog widget ---


def test_agent_log_add_decision():
    """AgentLog.add_decision should not raise."""
    from forge.widgets.agent_log import AgentLog
    log = AgentLog()
    # Just verify the method exists and is callable
    assert callable(log.add_decision)
    assert callable(log.add_notification)
    assert callable(log.add_agent_message)
    assert callable(log.add_system_event)
    assert callable(log.load_decisions)


# --- ForgeStatusBar widget ---


def test_status_bar_render_empty():
    from forge.widgets.status_bar import ForgeStatusBar
    bar = ForgeStatusBar()
    rendered = bar.render()
    assert rendered is not None


def test_status_bar_render_with_bindings():
    from forge.widgets.status_bar import ForgeStatusBar
    bar = ForgeStatusBar(bindings=[("P", "Pause"), ("E", "Eval")])
    rendered = bar.render()
    text = str(rendered)
    assert "P" in text
    assert "Pause" in text


# --- pyproject.toml entry point ---


def test_entry_point_points_to_main():
    """Verify pyproject.toml console_scripts points to forge.__main__:main."""
    import tomllib
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    scripts = data["project"]["scripts"]
    assert "forge" in scripts
    assert scripts["forge"] == "forge.__main__:main"


def test_textual_in_dependencies():
    """Verify textual is listed as a dependency, typer is not."""
    import tomllib
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = " ".join(data["project"]["dependencies"])
    assert "textual" in deps
    assert "typer" not in deps
