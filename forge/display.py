"""Rich terminal output utilities for auren-forge."""

from __future__ import annotations

from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Auren Research branded theme
FORGE_THEME = Theme(
    {
        "forge.title": "bold cyan",
        "forge.success": "bold green",
        "forge.warning": "bold yellow",
        "forge.error": "bold red",
        "forge.info": "dim white",
        "forge.metric": "bold white",
        "forge.label": "cyan",
        "forge.agent": "bold magenta",
    }
)

console = Console(theme=FORGE_THEME)


def print_banner() -> None:
    """Print the auren-forge ASCII banner."""
    banner = """
  ██████╗ ██╗   ██╗██████╗ ███████╗███╗   ██╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
 ██╔══██╗██║   ██║██╔══██╗██╔════╝████╗  ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
 ███████║██║   ██║██████╔╝█████╗  ██╔██╗ ██║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
 ██╔══██║██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
 ██║  ██║╚██████╔╝██║  ██║███████╗██║ ╚████║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    """
    console.print(Panel(
        Text(banner, style="bold cyan"),
        subtitle="[forge.info]AI-Orchestrated SFT Post-Training  •  Auren Research[/]",
        border_style="cyan",
        padding=(0, 2),
    ))


def print_success(msg: str) -> None:
    console.print(f"[forge.success]✓[/] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[forge.warning]⚠[/] {msg}")


def print_error(msg: str) -> None:
    console.print(f"[forge.error]✗[/] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[forge.info]  {msg}[/]")


def print_agent(msg: str) -> None:
    console.print(f"[forge.agent]🤖 Forge:[/] {msg}")


def make_training_progress() -> Progress:
    """Build a Rich Progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )


def print_status_table(state: Any) -> None:
    """Print a formatted status table from a compact state dict."""
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=False, padding=(0, 1))
    table.add_column("Key", style="forge.label", width=22)
    table.add_column("Value", style="forge.metric")

    status_color = {
        "training": "green",
        "paused": "yellow",
        "evaluating": "blue",
        "waiting_user": "magenta",
        "completed": "bright_green",
        "failed": "red",
        "configuring": "cyan",
    }

    s = state if isinstance(state, dict) else {}

    status = s.get("status", "unknown")
    color = status_color.get(status, "white")
    table.add_row("Status", f"[{color}]{status}[/]")
    table.add_row("Session ID", str(s.get("session_id", "")))
    table.add_row("Project", str(s.get("project", "")))
    table.add_row("Model", str(s.get("model", "")))
    table.add_row("Dataset", str(s.get("dataset", "")))
    table.add_row("Autonomy", str(s.get("autonomy", "")))
    table.add_row("Round", str(s.get("current_round", 1)))

    prog = s.get("progress", {})
    if prog:
        step = prog.get("step", 0)
        total = prog.get("total", 0)
        pct = f"{step / total * 100:.1f}%" if total > 0 else "N/A"
        table.add_row("Progress", f"{step}/{total} steps ({pct})")
        table.add_row("Epoch", str(prog.get("epoch", 0)))
        table.add_row("Elapsed", f"{prog.get('elapsed_min', 0):.1f} min")

    metrics = s.get("metrics", {})
    if metrics:
        loss = metrics.get("latest_loss")
        best = metrics.get("best_loss")
        trend = metrics.get("trend", "unknown")
        trend_icons = {
            "decreasing": "↓ [green]decreasing[/]",
            "increasing": "↑ [red]increasing[/]",
            "plateau": "→ [yellow]plateau[/]",
            "unstable": "~ [yellow]unstable[/]",
            "unknown": "? unknown",
        }
        if loss is not None:
            table.add_row("Latest Loss", f"{loss:.4f}")
        if best is not None:
            table.add_row("Best Loss", f"{best:.4f}")
        table.add_row("Trend", trend_icons.get(trend, trend))
        lr = metrics.get("lr")
        if lr is not None:
            table.add_row("Learning Rate", f"{lr:.2e}")

    pending = s.get("pending_user_action")
    if pending:
        table.add_row("[bold yellow]Awaiting User[/]", str(pending))

    console.print(Panel(table, title="[forge.title]Forge Training Status[/]", border_style="cyan"))


def print_decisions_log(decisions: list[dict[str, Any]], max_entries: int = 5) -> None:
    """Print recent agent decisions."""
    if not decisions:
        print_info("No agent decisions yet.")
        return

    table = Table(
        "Timestamp", "Trigger", "Action Taken",
        box=box.SIMPLE_HEAVY, border_style="magenta",
        show_lines=True,
    )
    for d in decisions[-max_entries:]:
        ts = str(d.get("ts", ""))[:19]
        trigger = str(d.get("trigger", ""))
        action = str(d.get("action", ""))[:80]
        table.add_row(ts, trigger, action)

    console.print(Panel(table, title="[forge.agent]Recent Agent Decisions[/]", border_style="magenta"))


def print_eval_results(eval_results: list[dict[str, Any]]) -> None:
    """Print evaluation results."""
    if not eval_results:
        print_info("No evaluation results yet.")
        return

    table = Table(
        "Round", "Avg Score", "Passed", "Details",
        box=box.ROUNDED, border_style="blue",
    )
    for r in eval_results:
        avg = r.get("avg_score", 0)
        passed = r.get("passed", False)
        passed_str = "[green]Yes[/]" if passed else "[red]No[/]"
        score_color = "green" if avg >= 3.5 else "yellow" if avg >= 2.5 else "red"
        table.add_row(
            str(r.get("round", "?")),
            f"[{score_color}]{avg:.2f}/5.0[/]",
            passed_str,
            "",
        )

    console.print(Panel(table, title="[bold blue]Evaluation History[/]", border_style="blue"))


def print_round_history(rounds: list[dict[str, Any]]) -> None:
    """Print the history of SFT rounds."""
    if not rounds:
        print_info("No completed rounds yet.")
        return

    table = Table(
        "Round", "Dataset", "Final Loss", "Eval Score", "Completed",
        box=box.ROUNDED, border_style="cyan",
    )
    for r in rounds:
        loss = r.get("final_loss")
        score = r.get("eval_score")
        table.add_row(
            str(r.get("round_num", "?")),
            str(r.get("dataset", "")),
            f"{loss:.4f}" if loss else "N/A",
            f"{score:.2f}" if score else "N/A",
            (r.get("completed_at") or "")[:19],
        )

    console.print(Panel(table, title="[forge.title]Training Round History[/]", border_style="cyan"))


def make_spinner(description: str = "Working...") -> Progress:
    """Return a simple spinner progress for short-running tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[forge.info]{description}"),
        console=console,
    )


def ask_confirm(question: str, default: bool = False) -> bool:
    """Ask a yes/no question and return the boolean answer."""
    hint = "[Y/n]" if default else "[y/N]"
    answer = console.input(f"[forge.label]{question} {hint}:[/] ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


def ask_input(prompt: str, default: Optional[str] = None) -> str:
    """Ask for a text input with an optional default."""
    hint = f" ([forge.info]{default}[/])" if default else ""
    result = console.input(f"[forge.label]{prompt}{hint}:[/] ").strip()
    return result if result else (default or "")
