"""Configuration models and YAML loading for auren-forge."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    name: str = "unsloth/Llama-3.1-8B-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True


class DatasetConfig(BaseModel):
    source: str
    format: str = "auto"  # auto | alpaca | sharegpt | custom
    split: str = "train"
    column_mapping: Optional[dict[str, str]] = None

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"auto", "alpaca", "sharegpt", "custom"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got '{v}'")
        return v


class TrainingConfig(BaseModel):
    lora_rank: int = 16
    lora_alpha: int = 16
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500
    output_dir: str = "./output"

    @field_validator("lora_rank", "lora_alpha")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class AgentConfig(BaseModel):
    provider: str = "openrouter"  # openrouter | ollama
    api_key: Optional[str] = None
    model: str = "xiaomi/mimo-v2-pro"
    base_url: Optional[str] = None  # for ollama
    autonomy: str = "suggest"  # monitor | suggest | auto
    check_interval_minutes: int = 30
    eval_on_completion: bool = True
    notification_method: str = "terminal"
    max_tokens_per_call: int = 4096
    temperature: float = 0.3

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openrouter", "ollama"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}")
        return v

    @field_validator("autonomy")
    @classmethod
    def validate_autonomy(cls, v: str) -> str:
        allowed = {"monitor", "suggest", "auto"}
        if v not in allowed:
            raise ValueError(f"autonomy must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def resolve_env_vars(self) -> "AgentConfig":
        """Expand ${ENV_VAR} references in api_key and base_url."""
        if self.api_key:
            self.api_key = _expand_env(self.api_key)
        if self.base_url:
            self.base_url = _expand_env(self.base_url)
        return self


class EvalConfig(BaseModel):
    num_test_prompts: int = 15
    min_quality_score: float = 3.0
    compare_with_previous: bool = True


class ForgeConfig(BaseModel):
    project_name: str = "my-sft-project"
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def _expand_env(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    pattern = re.compile(r"\$\{([^}]+)\}")
    def replacer(m: re.Match) -> str:
        var = m.group(1)
        result = os.environ.get(var)
        if result is None:
            return m.group(0)  # leave unexpanded if not set
        return result
    return pattern.sub(replacer, value)


def load_config(path: str | Path) -> ForgeConfig:
    """Load and validate a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty: {p}")
    return ForgeConfig.model_validate(raw)


def save_config(config: ForgeConfig, path: str | Path) -> None:
    """Save a ForgeConfig to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(exclude_none=True)
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def find_config(start_dir: str | Path | None = None) -> Path | None:
    """Search for forge.yaml or forge.yml in the current or parent directories."""
    search = Path(start_dir) if start_dir else Path.cwd()
    for candidate in [search, *search.parents]:
        for name in ("forge.yaml", "forge.yml"):
            p = candidate / name
            if p.exists():
                return p
    return None


# Recommended models shown during forge init
RECOMMENDED_MODELS = {
    "openrouter": [
        {
            "id": "anthropic/claude-opus-4.6",
            "label": "Claude Opus 4.6 (Primary — best judgment, production quality)",
            "cost": "~$5/M input, ~$25/M output",
        },
        {
            "id": "anthropic/claude-sonnet-4.6",
            "label": "Claude Sonnet 4.6 (Best balance of capability and cost)",
            "cost": "~$3/M input, ~$15/M output",
        },
        {
            "id": "xiaomi/mimo-v2-pro",
            "label": "MiMo v2 Pro (Budget — strong agent performance, 5x cheaper)",
            "cost": "~$1/M input, ~$3/M output",
        },
        {
            "id": "minimax/minimax-m2.7",
            "label": "MiniMax M2.7 (Very cheap, multi-agent planning)",
            "cost": "~$0.30/M input, ~$1.20/M output",
        },
        {
            "id": "z-ai/glm-5-turbo",
            "label": "GLM-5 Turbo (Fast inference, optimized for agent workflows)",
            "cost": "~$1.20/M input, ~$4/M output",
        },
    ],
    "ollama": [
        {"id": "minimax-m2.7:cloud", "label": "MiniMax M2.7 Cloud"},
        {"id": "glm-5:cloud", "label": "GLM-5 Cloud"},
        {"id": "kimi-k2.5:cloud", "label": "Kimi K2.5 Cloud (multimodal, Agent Swarm, free tier)"},
    ],
}
