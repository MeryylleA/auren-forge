"""Tests for forge.config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from forge.config import (
    AgentConfig,
    DatasetConfig,
    ForgeConfig,
    ModelConfig,
    TrainingConfig,
    _expand_env,
    find_config,
    load_config,
    save_config,
)


# --- _expand_env ---


def test_expand_env_set(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    assert _expand_env("${MY_KEY}") == "secret123"


def test_expand_env_unset():
    # unset var should stay as-is
    result = _expand_env("${NONEXISTENT_VAR_XYZ}")
    assert result == "${NONEXISTENT_VAR_XYZ}"


def test_expand_env_no_var():
    assert _expand_env("plain-string") == "plain-string"


# --- ModelConfig ---


def test_model_config_defaults():
    m = ModelConfig()
    assert m.name == "unsloth/Llama-3.1-8B-bnb-4bit"
    assert m.max_seq_length == 2048
    assert m.load_in_4bit is True


# --- DatasetConfig ---


def test_dataset_config_valid_formats():
    for fmt in ("auto", "alpaca", "sharegpt", "custom"):
        d = DatasetConfig(source="foo/bar", format=fmt)
        assert d.format == fmt


def test_dataset_config_invalid_format():
    with pytest.raises(Exception):
        DatasetConfig(source="foo/bar", format="badformat")


# --- TrainingConfig ---


def test_training_config_defaults():
    t = TrainingConfig()
    assert t.lora_rank == 16
    assert t.batch_size == 2
    assert t.learning_rate == 2e-4


def test_training_config_invalid_lora_rank():
    with pytest.raises(Exception):
        TrainingConfig(lora_rank=-1)


# --- AgentConfig ---


def test_agent_config_valid_providers():
    for p in ("openrouter", "ollama"):
        a = AgentConfig(provider=p)
        assert a.provider == p


def test_agent_config_invalid_provider():
    with pytest.raises(Exception):
        AgentConfig(provider="gpt4all")


def test_agent_config_valid_autonomy():
    for a in ("monitor", "suggest", "auto"):
        cfg = AgentConfig(autonomy=a)
        assert cfg.autonomy == a


def test_agent_config_env_expansion(monkeypatch):
    monkeypatch.setenv("OR_KEY", "mykey")
    a = AgentConfig(api_key="${OR_KEY}")
    assert a.api_key == "mykey"


# --- ForgeConfig ---


def test_forge_config_requires_dataset():
    with pytest.raises(Exception):
        # Missing dataset
        ForgeConfig()  # type: ignore[call-arg]


def test_forge_config_full():
    cfg = ForgeConfig(
        project_name="test-project",
        dataset=DatasetConfig(source="yahma/alpaca-cleaned"),
    )
    assert cfg.project_name == "test-project"
    assert cfg.model.name == "unsloth/Llama-3.1-8B-bnb-4bit"
    assert cfg.training.epochs == 1


# --- save_config / load_config ---


def test_save_and_load_config(tmp_path):
    cfg = ForgeConfig(
        project_name="roundtrip-test",
        dataset=DatasetConfig(source="test/dataset", format="alpaca"),
        training=TrainingConfig(epochs=3, lora_rank=32),
    )
    p = tmp_path / "forge.yaml"
    save_config(cfg, p)
    assert p.exists()

    loaded = load_config(p)
    assert loaded.project_name == "roundtrip-test"
    assert loaded.dataset.source == "test/dataset"
    assert loaded.training.epochs == 3
    assert loaded.training.lora_rank == 32


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.yaml")


# --- find_config ---


def test_find_config_found(tmp_path, monkeypatch):
    (tmp_path / "forge.yaml").write_text("project_name: x\ndataset:\n  source: foo\n")
    monkeypatch.chdir(tmp_path)
    result = find_config()
    assert result is not None
    assert result.name == "forge.yaml"


def test_find_config_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = find_config()
    assert result is None
