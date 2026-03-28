"""Typer CLI commands for auren-forge."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from forge import __version__
from forge.config import (
    RECOMMENDED_MODELS,
    AgentConfig,
    DatasetConfig,
    EvalConfig,
    ForgeConfig,
    ModelConfig,
    TrainingConfig,
    find_config,
    load_config,
    save_config,
)
from forge.display import (
    ask_confirm,
    ask_input,
    console,
    print_agent,
    print_banner,
    print_decisions_log,
    print_error,
    print_eval_results,
    print_info,
    print_round_history,
    print_status_table,
    print_success,
    print_warning,
)

app = typer.Typer(
    name="forge",
    help="auren-forge: AI-orchestrated supervised fine-tuning for language models.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "forge.yaml"


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


def _load_config_or_exit(config_path: Optional[str] = None) -> ForgeConfig:
    path = config_path or find_config()
    if path is None:
        print_error("No config file found. Run [bold]forge init[/bold] first.")
        raise typer.Exit(1)
    try:
        return load_config(path)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# forge init
# ---------------------------------------------------------------------------


@app.command("init")
def cmd_init(
    output: str = typer.Option(_DEFAULT_CONFIG, "--output", "-o", help="Config file path to write"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive setup wizard: configure model, dataset, and agent."""
    _setup_logging(verbose)
    print_banner()
    console.print("\n[bold cyan]Welcome to auren-forge setup![/bold cyan]\n")

    # Project name
    project_name = ask_input("Project name", default="my-sft-project")

    # Model
    console.print("\n[bold]Base model[/bold]")
    print_info("Any HuggingFace model supported by Unsloth. Popular 4-bit options:")
    for i, m in enumerate([
        "unsloth/Llama-3.1-8B-bnb-4bit",
        "unsloth/Llama-3.2-3B-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-bnb-4bit",
    ], 1):
        print_info(f"  {i}. {m}")
    model_name = ask_input("Model name (HuggingFace ID)", default="unsloth/Llama-3.1-8B-bnb-4bit")
    max_seq_length = int(ask_input("Max sequence length", default="2048"))

    # Dataset
    console.print("\n[bold]Training dataset[/bold]")
    print_info("HuggingFace hub ID (e.g. yahma/alpaca-cleaned) or local path.")
    dataset_source = ask_input("Dataset source")
    if not dataset_source:
        print_error("Dataset source is required.")
        raise typer.Exit(1)
    dataset_format = ask_input("Dataset format (auto/alpaca/sharegpt/custom)", default="auto")

    # Training hyperparameters
    console.print("\n[bold]Training hyperparameters[/bold]")
    epochs = int(ask_input("Epochs", default="1"))
    batch_size = int(ask_input("Batch size per device", default="2"))
    grad_accum = int(ask_input("Gradient accumulation steps", default="4"))
    lora_rank = int(ask_input("LoRA rank", default="16"))
    output_dir = ask_input("Output directory", default="./output")

    # Agent provider
    console.print("\n[bold]AI Agent configuration[/bold]")
    provider_choice = ask_input("Provider (openrouter/ollama)", default="openrouter")

    api_key = ""
    if provider_choice == "openrouter":
        console.print("\n[bold]Recommended OpenRouter models:[/bold]")
        for i, m in enumerate(RECOMMENDED_MODELS["openrouter"], 1):
            print_info(f"  {i}. {m['id']}")
            print_info(f"     {m['label']}")
            print_info(f"     Cost: {m['cost']}")
        agent_model = ask_input("Agent model", default=RECOMMENDED_MODELS["openrouter"][2]["id"])
        api_key = ask_input(
            "OpenRouter API key (or set OPENROUTER_API_KEY env var)",
            default="${OPENROUTER_API_KEY}",
        )
    else:
        console.print("\n[bold]Recommended Ollama Cloud models:[/bold]")
        for m in RECOMMENDED_MODELS["ollama"]:
            print_info(f"  • {m['id']} — {m['label']}")
        agent_model = ask_input("Agent model", default=RECOMMENDED_MODELS["ollama"][2]["id"])

    autonomy = ask_input("Agent autonomy level (monitor/suggest/auto)", default="suggest")
    check_interval = int(ask_input("Agent check interval (minutes)", default="30"))

    # Eval
    min_score = float(ask_input("Min quality score for eval (1-5)", default="3.0"))
    eval_on_completion = ask_confirm("Run eval automatically on training completion?", default=True)

    # Build config
    agent_cfg: dict = {
        "provider": provider_choice,
        "model": agent_model,
        "autonomy": autonomy,
        "check_interval_minutes": check_interval,
        "eval_on_completion": eval_on_completion,
    }
    if api_key:
        agent_cfg["api_key"] = api_key

    config = ForgeConfig(
        project_name=project_name,
        model=ModelConfig(name=model_name, max_seq_length=max_seq_length),
        dataset=DatasetConfig(source=dataset_source, format=dataset_format),
        training=TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            lora_rank=lora_rank,
            output_dir=output_dir,
        ),
        agent=AgentConfig(**agent_cfg),
        eval=EvalConfig(min_quality_score=min_score, eval_on_completion=eval_on_completion),
    )

    save_config(config, output)
    print_success(f"Config saved to [bold]{output}[/bold]")
    console.print(f"\n[bold]Next step:[/bold] [cyan]forge train --config {output}[/cyan]\n")


# ---------------------------------------------------------------------------
# forge train
# ---------------------------------------------------------------------------


@app.command("train")
def cmd_train(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing session"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start SFT training with AI agent monitoring."""
    _setup_logging(verbose)
    print_banner()

    config = _load_config_or_exit(config_path)
    tc = config.training
    output_dir = Path(tc.output_dir)

    from forge.session import SessionManager
    session_mgr = SessionManager(output_dir)

    # Check for existing session
    if session_mgr.exists() and not resume:
        state = session_mgr.load()
        if state.status in ("training", "paused", "waiting_user"):
            if ask_confirm(
                f"Found existing session (status={state.status}). Resume?", default=True
            ):
                resume = True
            else:
                if not ask_confirm("Start a new session? (will overwrite existing state)", default=False):
                    raise typer.Exit(0)

    if resume and session_mgr.exists():
        state, _ = session_mgr.load_or_create()
        print_info(f"Resumed session {state.session_id}")
    else:
        state = session_mgr.create_new(
            project_name=config.project_name,
            model_name=config.model.name,
            dataset_source=config.dataset.source,
            output_dir=str(output_dir),
            autonomy_level=config.agent.autonomy,
            config_path=str(config_path or _DEFAULT_CONFIG),
        )
        print_info(f"New session {state.session_id}")

    # Import training-only deps
    try:
        from forge.agent import AgentOrchestrator, build_provider
        from forge.callback import ForgeMonitorCallback
        from forge.dataset import DatasetHandler
        from forge.evaluation import Evaluator
        from forge.monitor import TrainingMonitor
        from forge.tools import ToolExecutor
        from forge.training import ForgeTrainer
    except ImportError as e:
        print_error(
            f"Training dependencies not installed: {e}\n"
            "Install with: pip install 'auren-forge[train]'"
        )
        raise typer.Exit(1)

    # Build subsystems
    trainer = ForgeTrainer(config=config, session=session_mgr)
    dataset_handler = DatasetHandler(
        model_name=config.model.name,
        column_mapping=config.dataset.column_mapping or {},
    )
    provider = build_provider(config.agent)
    executor = ToolExecutor(session=session_mgr)
    executor.set_trainer(trainer)
    executor.set_dataset_handler(dataset_handler)
    monitor = TrainingMonitor(
        session=session_mgr,
        check_interval_minutes=config.agent.check_interval_minutes,
    )
    agent = AgentOrchestrator(provider=provider, session=session_mgr, executor=executor)
    monitor.attach_agent(agent)

    # Attach evaluator after trainer loads model (done below)
    # We'll set it up after model load

    try:
        # Load model
        with console.status("[cyan]Loading model and applying LoRA...[/cyan]"):
            trainer.load_model()
        print_success(f"Model loaded: {config.model.name}")

        # Setup evaluator now that trainer has a model
        evaluator = Evaluator(
            provider=provider,
            trainer=trainer,
            session=session_mgr,
            min_quality_score=config.eval.min_quality_score,
        )
        executor.set_evaluator(evaluator)

        # Validate and load dataset
        console.print(f"\n[cyan]Validating dataset:[/cyan] {config.dataset.source}")
        validation = dataset_handler.validate(
            source=config.dataset.source,
            fmt=config.dataset.format,
            split=config.dataset.split,
        )
        if not validation["valid"]:
            print_warning("Dataset validation issues detected:")
            for issue in validation["issues"]:
                print_warning(f"  • {issue}")
            if not ask_confirm("Continue anyway?", default=False):
                raise typer.Exit(1)
        else:
            print_success(
                f"Dataset OK: {validation['num_rows']:,} rows, format={validation['format']}"
            )

        # Prepare dataset with chat template
        with console.status("[cyan]Formatting dataset with chat template...[/cyan]"):
            prepared_ds = dataset_handler.prepare_for_training(
                tokenizer=trainer.tokenizer,
                fmt=config.dataset.format if config.dataset.format != "auto" else validation["format"],
                max_seq_length=config.model.max_seq_length,
            )
        print_success(f"Dataset prepared: {len(prepared_ds):,} examples")

        # Build callback and trainer
        callback = ForgeMonitorCallback(session=session_mgr, monitor=monitor)
        trainer.build_trainer(dataset=prepared_ds, callback=callback)

        # Initial agent call
        agent.call_sync("training_start", {"config": config.model_dump(mode="json")})

        # Run training
        console.rule("[bold cyan]Training Started[/bold cyan]")
        if resume and state.last_checkpoint_path:
            print_info(f"Resuming from checkpoint: {state.last_checkpoint_path}")
            result = trainer.resume_from_checkpoint(state.last_checkpoint_path)
        else:
            result = trainer.train()

        session_mgr.set_status("completed" if not _was_paused(session_mgr) else "paused")

        # Post-training eval
        if config.agent.eval_on_completion and session_mgr.state.status == "completed":
            console.rule("[bold blue]Running Evaluation[/bold blue]")
            session_mgr.set_status("evaluating")
            eval_result = evaluator.run(num_prompts=config.eval.num_test_prompts)
            print_eval_result_summary(eval_result, config.eval.min_quality_score)

            if not eval_result.passed_threshold:
                print_warning(
                    f"Quality score {eval_result.avg_score:.2f} below threshold "
                    f"{config.eval.min_quality_score:.1f}."
                )
                print_agent(eval_result.summary)
                console.print(
                    "\n[bold]To run another SFT round:[/bold]\n"
                    "  1. Provide a new dataset: [cyan]forge train --config forge.yaml[/cyan]\n"
                    "  2. Or run: [cyan]forge resume[/cyan] after updating your config's dataset.\n"
                )
            else:
                print_success("Evaluation passed! Model quality meets threshold.")

            session_mgr.set_status("completed")

        monitor.shutdown()
        await_close(provider)

        console.rule("[bold green]Training Complete[/bold green]")
        print_success(f"Session {session_mgr.state.session_id} finished.")
        print_info(f"Model saved to: {tc.output_dir}")
        print_info("Export with: [cyan]forge export[/cyan]")

    except KeyboardInterrupt:
        print_warning("\nTraining interrupted by user.")
        session_mgr.set_status("paused")
        monitor.shutdown()
        await_close(provider)
    except Exception as e:
        print_error(f"Training failed: {e}")
        session_mgr.set_status("failed")
        monitor.shutdown()
        await_close(provider)
        logger.exception("Training error")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# forge status
# ---------------------------------------------------------------------------


@app.command("status")
def cmd_status(
    output_dir: str = typer.Option("./output", "--dir", "-d", help="Training output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show current session status, metrics, and agent decisions."""
    _setup_logging(verbose)
    from forge.session import SessionManager

    mgr = SessionManager(output_dir)
    if not mgr.exists():
        print_error(f"No session found in {output_dir}. Run [bold]forge train[/bold] first.")
        raise typer.Exit(1)

    state = mgr.load()
    compact = mgr.get_compact_state()
    print_status_table(compact)
    print_decisions_log(compact.get("last_decisions", []))
    print_eval_results(compact.get("eval_results", []))

    if state.pending_user_action:
        console.print(
            f"\n[bold yellow]Action Required:[/bold yellow] {state.pending_user_action}\n"
        )


# ---------------------------------------------------------------------------
# forge eval
# ---------------------------------------------------------------------------


@app.command("eval")
def cmd_eval(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    num_prompts: int = typer.Option(15, "--prompts", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Manually trigger evaluation of the current model checkpoint."""
    _setup_logging(verbose)
    config = _load_config_or_exit(config_path)

    from forge.agent import build_provider
    from forge.evaluation import Evaluator
    from forge.session import SessionManager
    from forge.training import ForgeTrainer

    mgr = SessionManager(config.training.output_dir)
    if not mgr.exists():
        print_error("No session found. Run forge train first.")
        raise typer.Exit(1)

    mgr.load()
    provider = build_provider(config.agent)

    with console.status("[cyan]Loading model for evaluation...[/cyan]"):
        trainer = ForgeTrainer(config=config, session=mgr)
        trainer.load_model()

    evaluator = Evaluator(
        provider=provider,
        trainer=trainer,
        session=mgr,
        min_quality_score=config.eval.min_quality_score,
    )

    console.rule("[bold blue]Running Evaluation[/bold blue]")
    mgr.set_status("evaluating")
    result = evaluator.run(num_prompts=num_prompts)
    print_eval_result_summary(result, config.eval.min_quality_score)
    mgr.set_status("completed")
    await_close(provider)


# ---------------------------------------------------------------------------
# forge resume
# ---------------------------------------------------------------------------


@app.command("resume")
def cmd_resume(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Resume an interrupted training session from state.json."""
    _setup_logging(verbose)
    # Delegate to train with --resume
    typer.echo("Resuming session...")
    cmd_train(config_path=config_path, resume=True, verbose=verbose)


# ---------------------------------------------------------------------------
# forge history
# ---------------------------------------------------------------------------


@app.command("history")
def cmd_history(
    output_dir: str = typer.Option("./output", "--dir", "-d"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show past training rounds and agent decisions."""
    _setup_logging(verbose)
    from forge.session import SessionManager

    mgr = SessionManager(output_dir)
    if not mgr.exists():
        print_error(f"No session found in {output_dir}.")
        raise typer.Exit(1)

    state = mgr.load()
    compact = mgr.get_compact_state()

    print_round_history([r.model_dump() for r in state.round_history])
    print_eval_results(compact.get("eval_results", []))
    print_decisions_log([d.model_dump() for d in state.decisions_log], max_entries=20)


# ---------------------------------------------------------------------------
# forge export
# ---------------------------------------------------------------------------


@app.command("export")
def cmd_export(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    fmt: str = typer.Option("safetensors", "--format", "-f", help="Export format: safetensors, hf, gguf"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Export the trained model (safetensors, hf, or gguf)."""
    _setup_logging(verbose)
    config = _load_config_or_exit(config_path)

    try:
        from forge.session import SessionManager
        from forge.training import ForgeTrainer
    except ImportError as e:
        print_error(f"Training dependencies missing: {e}")
        raise typer.Exit(1)

    mgr = SessionManager(config.training.output_dir)

    with console.status(f"[cyan]Loading model for export ({fmt})...[/cyan]"):
        trainer = ForgeTrainer(config=config, session=mgr)
        trainer.load_model()

    with console.status(f"[cyan]Exporting as {fmt}...[/cyan]"):
        exported_path = trainer.export(fmt=fmt, output_path=output_path)

    print_success(f"Model exported to: [bold]{exported_path}[/bold]")


# ---------------------------------------------------------------------------
# forge chat
# ---------------------------------------------------------------------------


@app.command("chat")
def cmd_chat(
    config_path: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Interactive chat with the Forge agent about the current session."""
    _setup_logging(verbose)
    config = _load_config_or_exit(config_path)

    from forge.agent import AgentOrchestrator, build_provider
    from forge.session import SessionManager
    from forge.tools import ToolExecutor

    mgr = SessionManager(config.training.output_dir)
    if not mgr.exists():
        print_error("No session found. Run forge train first.")
        raise typer.Exit(1)
    mgr.load()

    provider = build_provider(config.agent)
    executor = ToolExecutor(session=mgr)
    agent = AgentOrchestrator(provider=provider, session=mgr, executor=executor)

    console.print(
        Panel(
            "[forge.info]Type your message to chat with Forge agent.\n"
            "The agent has read-only access to session state in chat mode.\n"
            "Type [bold]exit[/bold] or [bold]quit[/bold] to leave.[/forge.info]",
            title="[forge.agent]Forge Chat[/forge.agent]",
            border_style="magenta",
        )
    )

    while True:
        try:
            user_input = console.input("[forge.label]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[forge.info]Bye![/forge.info]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[forge.info]Bye![/forge.info]")
            break

        if not user_input:
            continue

        with console.status("[forge.agent]Forge is thinking...[/forge.agent]"):
            result = agent.call_sync(
                trigger="user_chat",
                extra_context={"user_message": user_input},
            )

        notifications = executor.pop_notifications()
        for n in notifications:
            print_agent(n["message"])
        if not notifications:
            print_agent(result or "(no response)")

    await_close(provider)


# ---------------------------------------------------------------------------
# forge version
# ---------------------------------------------------------------------------


@app.command("version")
def cmd_version() -> None:
    """Show auren-forge version."""
    console.print(f"[bold cyan]auren-forge[/bold cyan] v{__version__}  •  Auren Research")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_eval_result_summary(result: Any, threshold: float) -> None:
    from rich.table import Table
    from rich import box

    table = Table(box=box.SIMPLE_HEAVY, border_style="blue", show_header=True)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="bold")
    for dim, score in result.scores.items():
        color = "green" if score >= threshold else "yellow" if score >= 2.5 else "red"
        table.add_row(dim.replace("_", " ").title(), f"[{color}]{score:.2f}/5.0[/]")

    avg_color = "green" if result.avg_score >= threshold else "red"
    table.add_row(
        "[bold]AVERAGE[/bold]",
        f"[{avg_color}][bold]{result.avg_score:.2f}/5.0[/bold][/]",
    )
    console.print(Panel(table, title="[bold blue]Evaluation Results[/]", border_style="blue"))
    print_agent(result.summary)


def _was_paused(mgr: Any) -> bool:
    return mgr.state.status == "paused"


def await_close(provider: Any) -> None:
    """Close the async provider client."""
    try:
        asyncio.run(provider.close())
    except Exception:
        pass


if __name__ == "__main__":
    app()
