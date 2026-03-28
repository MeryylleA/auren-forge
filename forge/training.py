"""Unsloth + SFTTrainer integration for auren-forge."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from forge.config import ForgeConfig, TrainingConfig
from forge.session import SessionManager

if TYPE_CHECKING:
    from forge.callback import ForgeMonitorCallback

logger = logging.getLogger(__name__)


class ForgeTrainer:
    """
    Wraps Unsloth + SFTTrainer.
    Handles model loading, LoRA application, training loop, checkpointing,
    and text generation for evaluation.
    """

    def __init__(self, config: ForgeConfig, session: SessionManager) -> None:
        self.config = config
        self.session = session
        self.model: Any = None
        self.tokenizer: Any = None
        self._trainer: Any = None
        self._callback: Optional["ForgeMonitorCallback"] = None

    def load_model(self) -> tuple[Any, Any]:
        """Load the base model and tokenizer via Unsloth."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError(
                "Unsloth is required for training. "
                "Install with: pip install unsloth (see https://github.com/unslothai/unsloth)"
            )

        mc = self.config.model
        logger.info("Loading model: %s", mc.name)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=mc.name,
            max_seq_length=mc.max_seq_length,
            load_in_4bit=mc.load_in_4bit,
        )

        tc = self.config.training
        logger.info("Applying LoRA (r=%d, alpha=%d)", tc.lora_rank, tc.lora_alpha)
        model = FastLanguageModel.get_peft_model(
            model,
            r=tc.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=tc.lora_alpha,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
        )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def build_trainer(
        self,
        dataset: Any,
        callback: "ForgeMonitorCallback",
    ) -> Any:
        """Build and configure SFTTrainer."""
        try:
            import torch
            from transformers import TrainingArguments
            from trl import SFTTrainer
        except ImportError as e:
            raise ImportError(f"Training dependencies missing: {e}. Run: pip install 'auren-forge[train]'") from e

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tc = self.config.training
        self._callback = callback

        # Determine mixed precision
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = torch.cuda.is_available() and not bf16

        output_dir = str(Path(tc.output_dir))
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            num_train_epochs=tc.epochs,
            learning_rate=tc.learning_rate,
            warmup_ratio=tc.warmup_ratio,
            fp16=fp16,
            bf16=bf16,
            logging_steps=tc.logging_steps,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=tc.save_steps,
            report_to="none",  # disable wandb/hub reporting by default
        )

        self._trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.model.max_seq_length,
            args=args,
            callbacks=[callback],
        )
        return self._trainer

    def train(self) -> Any:
        """Run the training loop. Returns trainer stats."""
        if self._trainer is None:
            raise RuntimeError("Trainer not built. Call build_trainer() first.")
        logger.info("Starting SFT training run")
        result = self._trainer.train()
        logger.info("Training complete. Final loss: %s", getattr(result, "training_loss", "N/A"))
        return result

    def resume_from_checkpoint(self, checkpoint_path: str) -> Any:
        """Resume training from a checkpoint."""
        if self._trainer is None:
            raise RuntimeError("Trainer not built. Call build_trainer() first.")
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        return self._trainer.train(resume_from_checkpoint=checkpoint_path)

    def generate(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        """Generate outputs for a list of prompts. Used by the evaluation system."""
        try:
            from unsloth import FastLanguageModel
            import torch
        except ImportError:
            raise ImportError("Unsloth and torch are required for generation.")

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        FastLanguageModel.for_inference(self.model)
        outputs: list[str] = []

        for prompt in prompts:
            inputs = self.tokenizer(
                [prompt], return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # Decode only the newly generated tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated[0][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text.strip())

        # Switch back to training mode
        FastLanguageModel.for_training(self.model)
        return outputs

    def save_checkpoint(self, label: str = "") -> str:
        """Save the current model checkpoint and return the path."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        tc = self.config.training
        suffix = f"-{label}" if label else ""
        step = self.session.state.training_progress.current_step
        path = str(Path(tc.output_dir) / f"checkpoint-manual{suffix}-{step}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Checkpoint saved: %s", path)
        return path

    def export(self, fmt: str = "safetensors", output_path: Optional[str] = None) -> str:
        """Export the trained model in the requested format."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        tc = self.config.training
        out = output_path or str(Path(tc.output_dir) / f"final-{fmt}")

        if fmt in ("safetensors", "hf"):
            self.model.save_pretrained(out)
            self.tokenizer.save_pretrained(out)
        elif fmt == "gguf":
            try:
                self.model.save_pretrained_gguf(out, self.tokenizer)
            except AttributeError:
                raise ValueError(
                    "GGUF export requires a newer version of Unsloth. "
                    "Update with: pip install --upgrade unsloth"
                )
        else:
            raise ValueError(f"Unknown export format: {fmt}. Use safetensors, hf, or gguf.")

        logger.info("Model exported to %s (format=%s)", out, fmt)
        return out

    def request_pause(self) -> None:
        if self._callback:
            self._callback.request_pause()

    def request_resume(self) -> None:
        if self._callback:
            self._callback.request_resume()
