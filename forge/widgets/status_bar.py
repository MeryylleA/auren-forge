"""Bottom status bar widget showing available hotkeys."""

from __future__ import annotations

from textual.widgets import Static

from rich.text import Text


class ForgeStatusBar(Static):
    """Bottom bar displaying contextual key bindings."""

    DEFAULT_CSS = """
    ForgeStatusBar {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: #888;
        content-align: center middle;
    }
    """

    def __init__(self, bindings: list[tuple[str, str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._bindings = bindings or []

    def set_bindings(self, bindings: list[tuple[str, str]]) -> None:
        self._bindings = bindings
        self.refresh()

    def render(self) -> Text:
        parts: list[Text] = []
        for key, label in self._bindings:
            parts.append(Text(f" {key} ", style="bold #b8860b on #333333"))
            parts.append(Text(f" {label} ", style="#aaaaaa"))
            parts.append(Text("  "))
        result = Text()
        for p in parts:
            result.append_text(p)
        return result
