"""Settings screen — global preferences saved to ~/.config/auren-forge/settings.yaml."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Header, Input, Label, RadioButton, RadioSet, Static, Switch

from forge.widgets.status_bar import ForgeStatusBar


class SettingsScreen(Screen):
    """Edit and save global auren-forge settings."""

    BINDINGS = [
        Binding("escape", "discard", "Cancel"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(" Settings", classes="section-header")
        with VerticalScroll(id="settings-container"):
            # Provider
            yield Static("Default Provider", classes="settings-label")
            with RadioSet(id="radio-default-provider"):
                yield RadioButton("OpenRouter", value=True, id="prov-openrouter")
                yield RadioButton("Ollama Cloud", id="prov-ollama")

            # OpenRouter
            yield Static("OpenRouter API Key", classes="settings-label")
            yield Input(
                placeholder="sk-or-v1-...",
                password=True,
                id="input-or-key",
            )

            # Ollama
            yield Static("Ollama Cloud Base URL", classes="settings-label")
            yield Input(
                value="https://api.ollama.com",
                placeholder="https://api.ollama.com",
                id="input-ollama-url",
            )

            # Default model
            yield Static("Default Agent Model", classes="settings-label")
            yield Input(
                value="xiaomi/mimo-v2-pro",
                placeholder="Model ID",
                id="input-default-model",
            )

            # Autonomy
            yield Static("Default Autonomy Level", classes="settings-label")
            with RadioSet(id="radio-default-autonomy"):
                yield RadioButton("Monitor only", id="auto-monitor")
                yield RadioButton("Suggest (recommend + wait)", value=True, id="auto-suggest")
                yield RadioButton("Auto (execute within guardrails)", id="auto-auto")

            # Notifications
            yield Static("Notifications", classes="settings-label")
            with Horizontal():
                yield Label("Terminal bell on events:  ")
                yield Switch(value=True, id="switch-bell")

            yield Label("")
            with Horizontal():
                yield Button("💾 Save", id="btn-save", variant="primary")
                yield Button("✗ Cancel", id="btn-cancel", variant="default")

        yield ForgeStatusBar(
            bindings=[("Ctrl+S", "Save"), ("Esc", "Cancel")]
        )

    def on_mount(self) -> None:
        self._load_settings()

    def _load_settings(self) -> None:
        settings = self.app.load_settings()
        if not settings:
            return

        provider = settings.get("default_provider", "openrouter")
        radio = self.query_one("#radio-default-provider", RadioSet)
        if provider == "ollama":
            radio.pressed_index = 1

        or_key = settings.get("openrouter_api_key", "")
        if or_key:
            self.query_one("#input-or-key", Input).value = or_key

        ollama_url = settings.get("ollama_base_url", "https://api.ollama.com")
        self.query_one("#input-ollama-url", Input).value = ollama_url

        model = settings.get("default_model", "xiaomi/mimo-v2-pro")
        self.query_one("#input-default-model", Input).value = model

        autonomy = settings.get("default_autonomy", "suggest")
        autonomy_radio = self.query_one("#radio-default-autonomy", RadioSet)
        idx = {"monitor": 0, "suggest": 1, "auto": 2}.get(autonomy, 1)
        autonomy_radio.pressed_index = idx

        bell = settings.get("terminal_bell", True)
        self.query_one("#switch-bell", Switch).value = bell

    def _collect_settings(self) -> dict:
        provider_idx = self.query_one("#radio-default-provider", RadioSet).pressed_index
        provider = "ollama" if provider_idx == 1 else "openrouter"

        autonomy_idx = self.query_one("#radio-default-autonomy", RadioSet).pressed_index
        autonomy = ["monitor", "suggest", "auto"][autonomy_idx] if autonomy_idx >= 0 else "suggest"

        return {
            "default_provider": provider,
            "openrouter_api_key": self.query_one("#input-or-key", Input).value.strip(),
            "ollama_base_url": self.query_one("#input-ollama-url", Input).value.strip(),
            "default_model": self.query_one("#input-default-model", Input).value.strip(),
            "default_autonomy": autonomy,
            "terminal_bell": self.query_one("#switch-bell", Switch).value,
        }

    def action_save(self) -> None:
        data = self._collect_settings()
        self.app.save_settings(data)
        self.notify("Settings saved to ~/.config/auren-forge/settings.yaml", severity="information")
        self.app.pop_screen()

    def action_discard(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_discard()
