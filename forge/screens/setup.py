"""Project Setup Wizard — multi-step configuration screen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
    Switch,
)

from forge.config import (
    RECOMMENDED_MODELS,
    AgentConfig,
    DatasetConfig,
    EvalConfig,
    ForgeConfig,
    ModelConfig,
    TrainingConfig,
    save_config,
)
from forge.widgets.status_bar import ForgeStatusBar

POPULAR_MODELS = [
    "unsloth/Llama-3.1-8B-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/Qwen2.5-7B-bnb-4bit",
]


class SetupScreen(Screen):
    """Interactive multi-step project setup wizard."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._step = 1
        self._total_steps = 4

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="setup-container"):
            # Step indicator
            yield Static("", id="step-indicator", classes="bold-gold")
            yield Label("")

            # --- Step 1: Model ---
            with Vertical(id="step-1", classes="setup-step"):
                yield Static("Base Model", classes="step-title")
                yield Label("Any HuggingFace model supported by Unsloth:")
                yield Input(
                    value="unsloth/Llama-3.1-8B-bnb-4bit",
                    placeholder="Model name or HuggingFace ID",
                    id="input-model",
                )
                yield Label("")
                yield Static("Popular choices:", classes="muted")
                for m in POPULAR_MODELS:
                    yield Static(f"  • {m}", classes="suggestion-list")
                yield Label("")
                yield Label("Max sequence length:")
                yield Input(value="2048", placeholder="2048", id="input-seq-len", type="integer")

            # --- Step 2: Dataset ---
            with Vertical(id="step-2", classes="setup-step"):
                yield Static("Training Dataset", classes="step-title")
                yield Label("HuggingFace hub ID or local path:")
                yield Input(
                    placeholder="e.g. yahma/alpaca-cleaned or ./data/train.jsonl",
                    id="input-dataset",
                )
                yield Label("")
                yield Label("Format:")
                with RadioSet(id="radio-format"):
                    yield RadioButton("Auto-detect", value=True, id="fmt-auto")
                    yield RadioButton("Alpaca", id="fmt-alpaca")
                    yield RadioButton("ShareGPT", id="fmt-sharegpt")
                    yield RadioButton("Custom", id="fmt-custom")
                yield Label("")
                yield Label("Split:")
                yield Input(value="train", placeholder="train", id="input-split")
                yield Label("")
                yield Button("Validate Dataset", id="btn-validate", variant="default")
                yield Static("", id="validation-result")

            # --- Step 3: Training ---
            with Vertical(id="step-3", classes="setup-step"):
                yield Static("Training Hyperparameters", classes="step-title")
                with Horizontal():
                    with Vertical():
                        yield Label("LoRA rank:")
                        yield Input(value="16", id="input-lora-rank", type="integer")
                    with Vertical():
                        yield Label("LoRA alpha:")
                        yield Input(value="16", id="input-lora-alpha", type="integer")
                with Horizontal():
                    with Vertical():
                        yield Label("Batch size:")
                        yield Input(value="2", id="input-batch", type="integer")
                    with Vertical():
                        yield Label("Grad accumulation:")
                        yield Input(value="4", id="input-grad-accum", type="integer")
                with Horizontal():
                    with Vertical():
                        yield Label("Epochs:")
                        yield Input(value="1", id="input-epochs", type="integer")
                    with Vertical():
                        yield Label("Learning rate:")
                        yield Input(value="2e-4", id="input-lr")
                yield Label("")
                yield Label("Output directory:")
                yield Input(value="./output", id="input-output-dir")

            # --- Step 4: Agent ---
            with Vertical(id="step-4", classes="setup-step"):
                yield Static("AI Agent Configuration", classes="step-title")
                yield Label("Provider:")
                with RadioSet(id="radio-provider"):
                    yield RadioButton("OpenRouter", value=True, id="prov-openrouter")
                    yield RadioButton("Ollama Cloud", id="prov-ollama")
                yield Label("")
                yield Label("API Key (or set OPENROUTER_API_KEY env var):")
                yield Input(
                    placeholder="sk-or-v1-...",
                    password=True,
                    id="input-api-key",
                )
                yield Label("")
                yield Label("Agent model:")
                yield Input(
                    value="xiaomi/mimo-v2-pro",
                    placeholder="Model ID",
                    id="input-agent-model",
                )
                yield Label("")
                yield Static("Recommended OpenRouter models:", classes="muted")
                for m in RECOMMENDED_MODELS["openrouter"]:
                    yield Static(
                        f"  {'★' if 'opus' in m['id'] else '◆'} {m['id']}  {m['cost']}",
                        classes="suggestion-list",
                    )
                yield Label("")
                yield Label("Autonomy level:")
                with RadioSet(id="radio-autonomy"):
                    yield RadioButton("Monitor only", id="auto-monitor")
                    yield RadioButton("Suggest (recommend + wait)", value=True, id="auto-suggest")
                    yield RadioButton("Auto (execute within guardrails)", id="auto-auto")
                yield Label("")
                with Horizontal():
                    with Vertical():
                        yield Label("Check interval (min):")
                        yield Input(value="30", id="input-interval", type="integer")
                    with Vertical():
                        yield Label("Min quality score (1-5):")
                        yield Input(value="3.0", id="input-min-score")
                yield Label("")
                with Horizontal():
                    yield Label("Eval on completion:  ")
                    yield Switch(value=True, id="switch-eval")

        # Navigation
        with Horizontal(id="setup-nav"):
            yield Button("← Back", id="btn-back", variant="default")
            yield Button("Next →", id="btn-next", variant="primary")
        yield ForgeStatusBar(bindings=[("Esc", "Cancel"), ("Tab", "Next field")])

    def on_mount(self) -> None:
        self._show_step(1)

    def _show_step(self, step: int) -> None:
        self._step = max(1, min(step, self._total_steps))
        for i in range(1, self._total_steps + 1):
            widget = self.query_one(f"#step-{i}")
            widget.display = i == self._step
        indicator = self.query_one("#step-indicator", Static)
        labels = {1: "Model", 2: "Dataset", 3: "Training", 4: "AI Agent"}
        indicator.update(f"━━━ Step {self._step}/{self._total_steps}: {labels[self._step]} ━━━")

        btn_next = self.query_one("#btn-next", Button)
        btn_back = self.query_one("#btn-back", Button)
        btn_back.disabled = self._step == 1
        if self._step == self._total_steps:
            btn_next.label = "🚀 Start Training"
            btn_next.variant = "success"
        else:
            btn_next.label = "Next →"
            btn_next.variant = "primary"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-next":
            if self._step < self._total_steps:
                self._show_step(self._step + 1)
            else:
                self._finish_setup()
        elif event.button.id == "btn-back":
            if self._step > 1:
                self._show_step(self._step - 1)
            else:
                self.action_go_back()
        elif event.button.id == "btn-validate":
            self._validate_dataset()

    def _validate_dataset(self) -> None:
        source = self.query_one("#input-dataset", Input).value.strip()
        result_widget = self.query_one("#validation-result", Static)
        if not source:
            result_widget.update("[red]Please enter a dataset source.[/red]")
            return
        result_widget.update("[dim]Validating...[/dim]")

        from forge.dataset import DatasetHandler
        handler = DatasetHandler()
        fmt = self._get_dataset_format()
        try:
            result = handler.validate(source=source, fmt=fmt)
            if result["valid"]:
                result_widget.update(
                    f"[green]✓ Valid: {result['format']} format, "
                    f"{result['num_rows']:,} samples, no issues[/green]"
                )
            else:
                issues = "; ".join(result["issues"][:2])
                result_widget.update(
                    f"[yellow]⚠ {result.get('num_rows', 0):,} samples, "
                    f"issues: {issues}[/yellow]"
                )
        except Exception as e:
            result_widget.update(f"[red]✗ Error: {e}[/red]")

    def _get_dataset_format(self) -> str:
        radio = self.query_one("#radio-format", RadioSet)
        idx = radio.pressed_index
        return ["auto", "alpaca", "sharegpt", "custom"][idx] if idx >= 0 else "auto"

    def _get_provider(self) -> str:
        radio = self.query_one("#radio-provider", RadioSet)
        return "ollama" if radio.pressed_index == 1 else "openrouter"

    def _get_autonomy(self) -> str:
        radio = self.query_one("#radio-autonomy", RadioSet)
        idx = radio.pressed_index
        return ["monitor", "suggest", "auto"][idx] if idx >= 0 else "suggest"

    def _finish_setup(self) -> None:
        """Build config from form values, save, and start training."""
        try:
            dataset_source = self.query_one("#input-dataset", Input).value.strip()
            if not dataset_source:
                self.notify("Dataset source is required.", severity="error")
                self._show_step(2)
                return

            provider = self._get_provider()
            api_key = self.query_one("#input-api-key", Input).value.strip()
            if provider == "openrouter" and not api_key:
                api_key = "${OPENROUTER_API_KEY}"

            config = ForgeConfig(
                project_name=_slugify(dataset_source),
                model=ModelConfig(
                    name=self.query_one("#input-model", Input).value.strip(),
                    max_seq_length=int(self.query_one("#input-seq-len", Input).value or 2048),
                ),
                dataset=DatasetConfig(
                    source=dataset_source,
                    format=self._get_dataset_format(),
                    split=self.query_one("#input-split", Input).value.strip() or "train",
                ),
                training=TrainingConfig(
                    lora_rank=int(self.query_one("#input-lora-rank", Input).value or 16),
                    lora_alpha=int(self.query_one("#input-lora-alpha", Input).value or 16),
                    batch_size=int(self.query_one("#input-batch", Input).value or 2),
                    gradient_accumulation_steps=int(self.query_one("#input-grad-accum", Input).value or 4),
                    epochs=int(self.query_one("#input-epochs", Input).value or 1),
                    learning_rate=float(self.query_one("#input-lr", Input).value or 2e-4),
                    output_dir=self.query_one("#input-output-dir", Input).value.strip() or "./output",
                ),
                agent=AgentConfig(
                    provider=provider,
                    api_key=api_key if api_key else None,
                    model=self.query_one("#input-agent-model", Input).value.strip(),
                    autonomy=self._get_autonomy(),
                    check_interval_minutes=int(self.query_one("#input-interval", Input).value or 30),
                    eval_on_completion=self.query_one("#switch-eval", Switch).value,
                ),
                eval=EvalConfig(
                    min_quality_score=float(self.query_one("#input-min-score", Input).value or 3.0),
                    eval_on_completion=self.query_one("#switch-eval", Switch).value,
                ),
            )

            config_path = Path("forge.yaml")
            save_config(config, config_path)
            self.app.config = config

            self.notify(f"Config saved to {config_path}", severity="information")

            from forge.screens.training import TrainingScreen
            self.app.switch_screen(
                TrainingScreen(config=config, config_path=str(config_path))
            )

        except Exception as e:
            self.notify(f"Setup error: {e}", severity="error")

    def action_go_back(self) -> None:
        self.app.pop_screen()


def _slugify(s: str) -> str:
    """Create a simple project name from a dataset source."""
    name = s.split("/")[-1].split(".")[0]
    return name.replace(" ", "-").lower()[:40]
