"""Live-updating training metrics panel widget."""

from __future__ import annotations

from typing import Any, Optional

from textual.widgets import Static

from rich.table import Table
from rich import box


class MetricsPanel(Static):
    """Displays current training metrics in a compact table."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._data: dict[str, Any] = {}

    def update_metrics(
        self,
        step: int = 0,
        total_steps: int = 0,
        epoch: float = 0.0,
        loss: Optional[float] = None,
        best_loss: Optional[float] = None,
        trend: str = "unknown",
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        elapsed_min: float = 0.0,
        eta_min: Optional[float] = None,
        round_num: int = 1,
        status: str = "training",
    ) -> None:
        self._data = {
            "step": step,
            "total_steps": total_steps,
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
            "trend": trend,
            "lr": lr,
            "grad_norm": grad_norm,
            "elapsed_min": elapsed_min,
            "eta_min": eta_min,
            "round_num": round_num,
            "status": status,
        }
        self.refresh()

    def render(self) -> Table:
        d = self._data
        if not d:
            return Table(title="Metrics", box=box.ROUNDED, border_style="#b8860b", expand=True)

        table = Table(box=box.ROUNDED, border_style="#b8860b", expand=True, show_header=False)
        table.add_column("Label", style="#aaa", width=16)
        table.add_column("Value", style="bold #f0e68c")

        step = d.get("step", 0)
        total = d.get("total_steps", 0)
        pct = f"  ({step / total * 100:.1f}%)" if total > 0 else ""
        table.add_row("Step", f"{step:,} / {total:,}{pct}")

        epoch = d.get("epoch", 0)
        table.add_row("Epoch", f"{epoch:.2f}")

        loss = d.get("loss")
        if loss is not None:
            table.add_row("Loss", f"{loss:.4f}")

        best = d.get("best_loss")
        if best is not None:
            table.add_row("Best Loss", f"{best:.4f}")

        trend = d.get("trend", "unknown")
        trend_display = {
            "decreasing": "[green]▼ decreasing[/]",
            "increasing": "[red]▲ increasing[/]",
            "plateau": "[yellow]► plateau[/]",
            "unstable": "[yellow]~ unstable[/]",
            "unknown": "[dim]? unknown[/]",
        }
        table.add_row("Trend", trend_display.get(trend, trend))

        lr = d.get("lr")
        if lr is not None:
            table.add_row("Learning Rate", f"{lr:.2e}")

        grad = d.get("grad_norm")
        if grad is not None:
            table.add_row("Grad Norm", f"{grad:.3f}")

        elapsed = d.get("elapsed_min", 0)
        table.add_row("Elapsed", f"{elapsed:.1f} min")

        eta = d.get("eta_min")
        if eta is not None:
            table.add_row("ETA", f"{eta:.1f} min")

        table.add_row("Round", str(d.get("round_num", 1)))

        status = d.get("status", "unknown")
        status_colors = {
            "training": "green",
            "paused": "yellow",
            "evaluating": "blue",
            "completed": "bright_green",
            "failed": "red",
            "waiting_user": "magenta",
        }
        color = status_colors.get(status, "white")
        table.add_row("Status", f"[{color}]{status}[/]")

        return table
