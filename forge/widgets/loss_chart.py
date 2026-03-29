"""ASCII loss curve widget for live training display."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from rich.text import Text


class LossChart(Static):
    """Renders an ASCII loss chart using Unicode block characters.

    Updates reactively when loss_values changes.
    """

    BLOCKS = " ▁▂▃▄▅▆▇█"

    loss_values: reactive[list[float]] = reactive(list, layout=True)

    def __init__(
        self,
        chart_width: int = 60,
        chart_height: int = 8,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.chart_width = chart_width
        self.chart_height = chart_height

    def watch_loss_values(self) -> None:
        self.refresh()

    def render(self) -> Text:
        vals = list(self.loss_values)
        if not vals:
            return Text("  No loss data yet — waiting for training steps...", style="dim")

        # Take the last `chart_width` values
        display = vals[-self.chart_width :]
        mn = min(display)
        mx = max(display)
        rng = mx - mn if mx > mn else 1.0

        lines: list[str] = []

        # Multi-row chart: each row represents a slice of the value range
        for row in range(self.chart_height - 1, -1, -1):
            row_low = mn + (rng * row / self.chart_height)
            row_high = mn + (rng * (row + 1) / self.chart_height)
            chars: list[str] = []
            for v in display:
                if v >= row_high:
                    chars.append("█")
                elif v >= row_low:
                    frac = (v - row_low) / (row_high - row_low) if row_high > row_low else 0
                    idx = min(8, int(frac * 8))
                    chars.append(self.BLOCKS[idx])
                else:
                    chars.append(" ")
            # Y-axis label
            label = f"{row_high:>6.3f}│" if row % 2 == 0 or self.chart_height <= 4 else "       │"
            lines.append(f"{label}{''.join(chars)}")

        # X-axis
        x_label = f"{mn:>6.3f}└{'─' * len(display)}"
        lines.append(x_label)

        # Legend
        latest = vals[-1]
        best = min(vals)
        legend = f"  latest: {latest:.4f}  best: {best:.4f}  points: {len(vals)}"
        lines.append(legend)

        return Text("\n".join(lines), style="#daa520")

    def update_values(self, values: list[float]) -> None:
        """Push new values into the chart."""
        self.loss_values = values

    def append_value(self, value: float) -> None:
        """Append a single value and refresh."""
        current = list(self.loss_values)
        current.append(value)
        # Keep bounded
        if len(current) > 2000:
            current = current[-2000:]
        self.loss_values = current
