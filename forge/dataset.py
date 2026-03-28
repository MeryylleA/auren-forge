"""Dataset loading, format detection, validation, and chat template mapping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Supported dataset formats and their required/optional columns
FORMAT_SIGNATURES: dict[str, dict[str, list[str]]] = {
    "alpaca": {
        "required": ["instruction", "output"],
        "optional": ["input"],
    },
    "sharegpt": {
        "required": ["conversations"],
        "optional": [],
    },
}


class DatasetInfo:
    def __init__(
        self,
        source: str,
        fmt: str,
        num_rows: int,
        columns: list[str],
        issues: list[str],
        dataset: Any = None,
    ) -> None:
        self.source = source
        self.format = fmt
        self.num_rows = num_rows
        self.columns = columns
        self.issues = issues
        self.dataset = dataset  # the raw HF dataset object, if loaded

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format": self.format,
            "num_rows": self.num_rows,
            "columns": self.columns,
            "issues": self.issues,
            "valid": self.is_valid,
        }


class DatasetHandler:
    """Handles loading, format detection, and validation of training datasets."""

    def __init__(
        self,
        model_name: str = "",
        column_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.column_mapping = column_mapping or {}
        self._loaded_dataset: Any = None
        self._loaded_info: Optional[DatasetInfo] = None

    def load(self, source: str, fmt: str = "auto", split: str = "train") -> dict[str, Any]:
        """Load a dataset and return basic info."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for training. "
                "Install with: pip install datasets"
            )

        logger.info("Loading dataset: %s (format=%s, split=%s)", source, fmt, split)

        # Local path vs HuggingFace hub
        p = Path(source)
        if p.exists():
            if source.endswith(".json") or source.endswith(".jsonl"):
                ds = load_dataset("json", data_files=source, split=split)
            elif source.endswith(".csv"):
                ds = load_dataset("csv", data_files=source, split=split)
            elif p.is_dir():
                ds = load_dataset(source, split=split)
            else:
                ds = load_dataset(source, split=split)
        else:
            ds = load_dataset(source, split=split)

        detected_fmt = fmt if fmt != "auto" else self._detect_format(list(ds.column_names))
        self._loaded_dataset = ds
        info = DatasetInfo(
            source=source,
            fmt=detected_fmt,
            num_rows=len(ds),
            columns=list(ds.column_names),
            issues=[],
        )
        info.dataset = ds
        self._loaded_info = info
        logger.info("Loaded %d rows, format=%s", len(ds), detected_fmt)
        return info.to_dict()

    def validate(self, source: str, fmt: str = "auto", split: str = "train") -> dict[str, Any]:
        """Load and validate a dataset. Returns a validation report."""
        try:
            info_dict = self.load(source=source, fmt=fmt, split=split)
        except Exception as e:
            return {"valid": False, "issues": [str(e)], "source": source}

        issues: list[str] = []
        ds = self._loaded_dataset
        detected_fmt = info_dict["format"]

        if detected_fmt == "unknown":
            issues.append(
                f"Could not detect dataset format. Columns: {info_dict['columns']}. "
                "Expected: alpaca (instruction/output), sharegpt (conversations), or specify --format."
            )
        elif detected_fmt in FORMAT_SIGNATURES:
            sig = FORMAT_SIGNATURES[detected_fmt]
            for col in sig["required"]:
                mapped = self.column_mapping.get(col, col)
                if mapped not in info_dict["columns"]:
                    issues.append(f"Missing required column '{mapped}' for {detected_fmt} format.")

        # Check for empty rows
        if ds is not None and detected_fmt != "unknown":
            empty_count = 0
            sample_size = min(500, len(ds))
            for i in range(sample_size):
                row = ds[i]
                if detected_fmt == "alpaca":
                    inst_col = self.column_mapping.get("instruction", "instruction")
                    out_col = self.column_mapping.get("output", "output")
                    if inst_col in row and not str(row.get(inst_col, "")).strip():
                        empty_count += 1
                    elif out_col in row and not str(row.get(out_col, "")).strip():
                        empty_count += 1
                elif detected_fmt == "sharegpt":
                    convs = row.get("conversations", [])
                    if not convs:
                        empty_count += 1
            if empty_count > 0:
                pct = round(empty_count / sample_size * 100, 1)
                issues.append(
                    f"{empty_count}/{sample_size} sampled rows have empty required fields ({pct}%)."
                )

        if info_dict["num_rows"] < 10:
            issues.append(
                f"Dataset is very small ({info_dict['num_rows']} rows). "
                "At least 100 rows recommended for meaningful SFT."
            )

        result = {**info_dict, "valid": len(issues) == 0, "issues": issues}
        if issues:
            logger.warning("Dataset validation issues: %s", issues)
        else:
            logger.info("Dataset validation passed for %s", source)
        return result

    def prepare_for_training(
        self,
        tokenizer: Any,
        fmt: Optional[str] = None,
        max_seq_length: int = 2048,
    ) -> Any:
        """Format the loaded dataset and apply the chat template."""
        if self._loaded_dataset is None:
            raise RuntimeError("No dataset loaded. Call load() first.")

        effective_fmt = fmt or (self._loaded_info.format if self._loaded_info else "unknown")
        ds = self._loaded_dataset

        if effective_fmt == "alpaca":
            ds = ds.map(
                lambda row: _format_alpaca(row, tokenizer, self.column_mapping, max_seq_length),
                batched=False,
                remove_columns=ds.column_names,
            )
        elif effective_fmt == "sharegpt":
            ds = ds.map(
                lambda row: _format_sharegpt(row, tokenizer, max_seq_length),
                batched=False,
                remove_columns=ds.column_names,
            )
        else:
            raise ValueError(
                f"Cannot prepare dataset with unknown format '{effective_fmt}'. "
                "Specify format explicitly."
            )

        logger.info("Prepared %d training examples", len(ds))
        return ds

    def _detect_format(self, columns: list[str]) -> str:
        """Detect dataset format from column names."""
        col_set = set(columns)
        if "conversations" in col_set:
            return "sharegpt"
        if "instruction" in col_set and "output" in col_set:
            return "alpaca"
        # Check for mapped columns
        for fmt, sig in FORMAT_SIGNATURES.items():
            required = sig["required"]
            if all(
                self.column_mapping.get(c, c) in col_set for c in required
            ):
                return fmt
        return "unknown"


def _format_alpaca(
    row: dict[str, Any],
    tokenizer: Any,
    column_mapping: dict[str, str],
    max_seq_length: int,
) -> dict[str, Any]:
    inst_col = column_mapping.get("instruction", "instruction")
    inp_col = column_mapping.get("input", "input")
    out_col = column_mapping.get("output", "output")

    instruction = str(row.get(inst_col, ""))
    inp = str(row.get(inp_col, "")) if inp_col in row else ""
    output = str(row.get(out_col, ""))

    if inp:
        user_content = f"{instruction}\n\n{inp}"
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    text = _apply_chat_template(tokenizer, messages)
    return {"text": text}


def _format_sharegpt(
    row: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    raw_convs = row.get("conversations", [])
    messages = []
    for turn in raw_convs:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        # Normalize sharegpt role names
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        messages.append({"role": role, "content": content})

    text = _apply_chat_template(tokenizer, messages)
    return {"text": text}


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the tokenizer's chat template, falling back gracefully."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple concatenation
        parts = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n".join(parts) + "\n"
