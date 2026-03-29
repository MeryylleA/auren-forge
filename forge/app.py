"""Main Textual TUI application for auren-forge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding

from forge import __version__

logger = logging.getLogger(__name__)

# XDG-compliant settings path
SETTINGS_DIR = Path.home() / ".config" / "auren-forge"
SETTINGS_PATH = SETTINGS_DIR / "settings.yaml"


class ForgeApp(App):
    """Auren Forge — AI-Orchestrated SFT Post-Training."""

    TITLE = "AUREN FORGE"
    SUB_TITLE = f"AI-Orchestrated SFT  v{__version__}"
    CSS_PATH = "forge.tcss"

    BINDINGS = [
        Binding("q", "request_quit", "Quit", show=True, priority=True),
        Binding("d", "toggle_dark", "Dark/Light", show=True),
    ]

    # Shared application state — screens read/write these
    config: Any = None
    session_manager: Any = None
    trainer: Any = None
    provider: Any = None
    agent: Any = None
    executor: Any = None
    monitor: Any = None

    def on_mount(self) -> None:
        _setup_logging()
        from forge.screens.home import HomeScreen
        self.install_screen(HomeScreen(), name="home")
        self.push_screen("home")

    def action_toggle_dark(self) -> None:
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"

    def action_request_quit(self) -> None:
        if self.monitor:
            self.monitor.shutdown()
        self.exit()

    # --- Helpers for screens ---

    def load_settings(self) -> dict[str, Any]:
        """Load global settings from ~/.config/auren-forge/settings.yaml."""
        import yaml
        if SETTINGS_PATH.exists():
            with open(SETTINGS_PATH) as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_settings(self, data: dict[str, Any]) -> None:
        """Save global settings to ~/.config/auren-forge/settings.yaml."""
        import yaml
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def find_recent_sessions(self, search_dirs: list[str] | None = None) -> list[dict[str, Any]]:
        """Scan for state.json files and return session summaries."""
        import json
        dirs = search_dirs or ["./output"]
        sessions: list[dict[str, Any]] = []
        for d in dirs:
            p = Path(d)
            if not p.exists():
                continue
            # Check direct state.json
            state_file = p / "state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data = json.load(f)
                    sessions.append(data)
                except Exception:
                    pass
            # Check subdirectories
            for sub in p.iterdir():
                if sub.is_dir():
                    sf = sub / "state.json"
                    if sf.exists():
                        try:
                            with open(sf) as f:
                                data = json.load(f)
                            sessions.append(data)
                        except Exception:
                            pass
        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("forge.log", mode="a")],
    )
