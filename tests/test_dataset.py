"""Tests for forge.dataset module."""

import pytest

from forge.dataset import DatasetHandler, _apply_chat_template


# Helper stub tokenizer that just joins messages
class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


# --- format detection helper ---


def _detect_format_from_columns(columns, column_mapping=None):
    handler = DatasetHandler(column_mapping=column_mapping or {})
    return handler._detect_format(columns)


def test_detect_alpaca():
    cols = ["instruction", "input", "output"]
    assert _detect_format_from_columns(cols) == "alpaca"


def test_detect_sharegpt():
    cols = ["conversations", "id"]
    assert _detect_format_from_columns(cols) == "sharegpt"


def test_detect_unknown():
    cols = ["question", "answer", "category"]
    result = _detect_format_from_columns(cols)
    assert result == "unknown"


def test_detect_alpaca_with_mapping():
    cols = ["question", "answer"]
    result = _detect_format_from_columns(
        cols, column_mapping={"instruction": "question", "output": "answer"}
    )
    assert result == "alpaca"


# --- _apply_chat_template ---


def test_apply_chat_template_basic():
    tokenizer = StubTokenizer()
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = _apply_chat_template(tokenizer, messages)
    assert "user" in result
    assert "Hello" in result
    assert "Hi there" in result


def test_apply_chat_template_fallback():
    """When tokenizer raises, fallback should work."""
    class BrokenTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("not supported")

    messages = [
        {"role": "user", "content": "Test"},
        {"role": "assistant", "content": "Response"},
    ]
    result = _apply_chat_template(BrokenTokenizer(), messages)
    assert "Test" in result
    assert "Response" in result


# --- DatasetHandler.validate (without loading real data) ---


def test_validate_missing_source(tmp_path):
    handler = DatasetHandler()
    result = handler.validate(source=str(tmp_path / "nonexistent.json"))
    assert result["valid"] is False
    assert len(result["issues"]) > 0


# --- DatasetHandler format detection ---


def test_handler_detect_alpaca():
    handler = DatasetHandler()
    assert handler._detect_format(["instruction", "input", "output"]) == "alpaca"


def test_handler_detect_sharegpt():
    handler = DatasetHandler()
    assert handler._detect_format(["conversations"]) == "sharegpt"


def test_handler_detect_unknown():
    handler = DatasetHandler()
    assert handler._detect_format(["col1", "col2"]) == "unknown"


# --- Column mapping ---


def test_column_mapping_in_handler():
    handler = DatasetHandler(column_mapping={"instruction": "q", "output": "a"})
    assert handler._detect_format(["q", "a"]) == "alpaca"
