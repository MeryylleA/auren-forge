"""Tool definitions and implementations callable by the AI agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from forge.session import SessionManager

logger = logging.getLogger(__name__)

# --- OpenAI-compatible tool schemas ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_training_status",
            "description": (
                "Get current training progress including step, loss, learning rate, "
                "and estimated time remaining."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loss_history",
            "description": "Get loss values for the last N training steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n_steps": {
                        "type": "integer",
                        "description": "Number of recent steps to retrieve",
                        "default": 100,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pause_training",
            "description": "Pause the current training run. Use when anomalies are detected.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resume_training",
            "description": "Resume a paused training run.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eval",
            "description": "Run evaluation on the current model checkpoint with test prompts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_prompts": {
                        "type": "integer",
                        "default": 15,
                        "description": "Number of test prompts to generate and evaluate",
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional categories to focus evaluation on",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_outputs",
            "description": (
                "Generate outputs from the current model checkpoint for given prompts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of prompts to run through the model",
                    }
                },
                "required": ["prompts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": (
                "Send a message to the user. Use for status updates, "
                "recommendations, or requesting input."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "requires_response": {"type": "boolean", "default": False},
                    "priority": {
                        "type": "string",
                        "enum": ["info", "warning", "critical"],
                        "default": "info",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_dataset",
            "description": (
                "Load and validate a dataset, checking format, completeness, "
                "and compatibility with the current model's chat template."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "HuggingFace dataset ID or local path",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["auto", "alpaca", "sharegpt", "custom"],
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_summary",
            "description": (
                "Get a compact summary of the entire session including all rounds, "
                "decisions, and current status."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_checkpoint",
            "description": "Save the current model checkpoint to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Optional label for the checkpoint",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": "Load a dataset and prepare it for the next training round.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "HuggingFace dataset ID or local path",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["auto", "alpaca", "sharegpt", "custom"],
                        "default": "auto",
                    },
                },
                "required": ["source"],
            },
        },
    },
]


# --- Tool executor ---


class ToolExecutor:
    """Dispatches agent tool calls to their Python implementations."""

    def __init__(self, session: "SessionManager") -> None:
        self.session = session
        # These are injected after the relevant subsystems are ready
        self._trainer: Any = None
        self._evaluator: Any = None
        self._dataset_handler: Any = None
        self._notification_queue: list[dict[str, Any]] = []

    def set_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def set_evaluator(self, evaluator: Any) -> None:
        self._evaluator = evaluator

    def set_dataset_handler(self, handler: Any) -> None:
        self._dataset_handler = handler

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call and return a JSON-serializable result."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return handler(**arguments)
        except Exception as e:
            logger.exception("Tool %s raised an error: %s", name, e)
            return {"error": str(e)}

    # --- Tool implementations ---

    def _tool_get_training_status(self) -> dict[str, Any]:
        state = self.session.state
        prog = state.training_progress
        metrics = state.metrics_summary
        return {
            "status": state.status,
            "step": prog.current_step,
            "total_steps": prog.total_steps,
            "progress_pct": (
                round(prog.current_step / prog.total_steps * 100, 1)
                if prog.total_steps > 0
                else 0
            ),
            "epoch": round(prog.current_epoch, 3),
            "elapsed_min": round(prog.elapsed_seconds / 60, 1),
            "estimated_remaining_min": (
                round(prog.estimated_remaining_seconds / 60, 1)
                if prog.estimated_remaining_seconds is not None
                else None
            ),
            "latest_loss": metrics.latest_loss,
            "best_loss": metrics.best_loss,
            "trend": metrics.trend,
            "learning_rate": metrics.learning_rate,
        }

    def _tool_get_loss_history(self, last_n_steps: int = 100) -> dict[str, Any]:
        history = self.session.get_recent_loss_history(last_n_steps)
        return {"count": len(history), "history": history}

    def _tool_pause_training(self) -> dict[str, Any]:
        if self._trainer is None:
            return {"error": "Trainer not attached"}
        self._trainer.request_pause()
        self.session.set_status("paused")
        logger.info("Training paused by agent")
        return {"ok": True, "message": "Training pause requested"}

    def _tool_resume_training(self) -> dict[str, Any]:
        if self._trainer is None:
            return {"error": "Trainer not attached"}
        self._trainer.request_resume()
        self.session.set_status("training")
        logger.info("Training resumed by agent")
        return {"ok": True, "message": "Training resume requested"}

    def _tool_run_eval(
        self,
        num_prompts: int = 15,
        categories: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        if self._evaluator is None:
            return {"error": "Evaluator not attached — evaluation not available yet"}
        self.session.set_status("evaluating")
        try:
            result = self._evaluator.run(num_prompts=num_prompts, categories=categories or [])
            return {"ok": True, "avg_score": result.avg_score, "summary": result.summary}
        except Exception as e:
            self.session.set_status("training")
            return {"error": str(e)}
        finally:
            if self.session.state.status == "evaluating":
                self.session.set_status("training")

    def _tool_get_model_outputs(self, prompts: list[str]) -> dict[str, Any]:
        if self._trainer is None:
            return {"error": "Trainer not attached"}
        if not prompts:
            return {"error": "No prompts provided"}
        try:
            outputs = self._trainer.generate(prompts)
            return {"outputs": [{"prompt": p, "response": r} for p, r in zip(prompts, outputs)]}
        except Exception as e:
            return {"error": str(e)}

    def _tool_send_notification(
        self,
        message: str,
        requires_response: bool = False,
        priority: str = "info",
    ) -> dict[str, Any]:
        entry = {
            "message": message,
            "requires_response": requires_response,
            "priority": priority,
        }
        self._notification_queue.append(entry)
        logger.info("[%s] Agent notification: %s", priority.upper(), message)
        if requires_response:
            self.session.set_pending_user_action(message)
        return {"ok": True, "queued": True}

    def _tool_validate_dataset(
        self,
        source: str,
        format: str = "auto",
    ) -> dict[str, Any]:
        if self._dataset_handler is None:
            return {"error": "Dataset handler not attached"}
        try:
            result = self._dataset_handler.validate(source=source, fmt=format)
            return result
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_session_summary(self) -> dict[str, Any]:
        return self.session.get_compact_state()

    def _tool_save_checkpoint(self, label: str = "") -> dict[str, Any]:
        if self._trainer is None:
            return {"error": "Trainer not attached"}
        try:
            path = self._trainer.save_checkpoint(label=label)
            self.session.set_checkpoint(path)
            return {"ok": True, "path": path}
        except Exception as e:
            return {"error": str(e)}

    def _tool_load_dataset(
        self,
        source: str,
        format: str = "auto",
    ) -> dict[str, Any]:
        if self._dataset_handler is None:
            return {"error": "Dataset handler not attached"}
        try:
            info = self._dataset_handler.load(source=source, fmt=format)
            return {"ok": True, "source": source, "num_rows": info.get("num_rows"), "format": info.get("format")}
        except Exception as e:
            return {"error": str(e)}

    def pop_notifications(self) -> list[dict[str, Any]]:
        """Drain and return queued notifications."""
        notifications = list(self._notification_queue)
        self._notification_queue.clear()
        return notifications
