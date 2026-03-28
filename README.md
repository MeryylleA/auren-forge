# auren-forge

**AI-orchestrated supervised fine-tuning for language models**

auren-forge is an open-source CLI tool by Auren Research that automates SFT post-training of language models using an AI agent as orchestrator. The agent monitors training in real-time, runs evaluations, analyzes output quality, and coordinates with the user when intervention is needed.

---

## Features

- **AI-monitored training** — an LLM agent watches loss curves, detects anomalies (NaN, spikes, plateaus, divergence), and takes action based on your configured autonomy level
- **Stateless, crash-safe architecture** — all context lives in `state.json`; the agent is called on-demand, not in a persistent session
- **Three autonomy levels** — `monitor` (observe only), `suggest` (recommend + wait), `auto` (execute within guardrails)
- **Multi-round SFT** — if quality is below threshold, the agent notifies you and recommends additional data; new training rounds always require explicit user approval
- **Universal model support** — any HuggingFace model supported by Unsloth
- **Universal dataset support** — Alpaca, ShareGPT, or custom column mapping
- **Multi-provider agent** — OpenRouter (recommended) or Ollama Cloud
- **Rich terminal UI** — live progress bars, status tables, decision logs

---

## Installation

```bash
# Base install (CLI + config tools, no GPU required)
pip install auren-forge

# Full install with training dependencies (requires CUDA)
pip install "auren-forge[train]"

# Development install
git clone https://github.com/meryyllea/auren-forge
cd auren-forge
pip install -e ".[train,dev]"
```

---

## Quick Start

```bash
# 1. Initialize a project
forge init

# 2. Start training with AI agent monitoring
forge train --config forge.yaml

# 3. Check status while training
forge status

# 4. Chat with the agent about the session
forge chat

# 5. Export the trained model
forge export --format safetensors
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `forge init` | Interactive setup wizard |
| `forge train [--config FILE]` | Start SFT training with agent monitoring |
| `forge status` | Show current session status, metrics, decisions |
| `forge eval` | Manually trigger evaluation |
| `forge chat` | Chat with the agent about the current session |
| `forge resume` | Resume an interrupted session |
| `forge history` | Show past training rounds and decisions |
| `forge export [--format gguf\|safetensors\|hf]` | Export trained model |
| `forge version` | Show version |

---

## Configuration

Run `forge init` for an interactive wizard, or copy and edit `configs/default.yaml`.

### Key sections

```yaml
model:
  name: "unsloth/Llama-3.1-8B-bnb-4bit"   # any Unsloth-supported HF model
  max_seq_length: 2048

dataset:
  source: "yahma/alpaca-cleaned"            # HF hub ID or local path
  format: "auto"                             # auto | alpaca | sharegpt | custom

training:
  lora_rank: 16
  epochs: 1
  learning_rate: 2e-4
  output_dir: "./output"

agent:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "xiaomi/mimo-v2-pro"
  autonomy: "suggest"                        # monitor | suggest | auto
  check_interval_minutes: 30
  eval_on_completion: true

eval:
  num_test_prompts: 15
  min_quality_score: 3.0                     # 1–5 scale
```

### Environment variable expansion

API keys can be stored as environment variables:
```yaml
api_key: "${OPENROUTER_API_KEY}"
```

---

## Supported Agent Models

### OpenRouter

| Model | Notes | Cost |
|-------|-------|------|
| `anthropic/claude-opus-4.6` | **Primary recommendation.** Best judgment, 1M context, deep problem decomposition | ~$5/M in, ~$25/M out |
| `anthropic/claude-sonnet-4.6` | Best balance of capability and cost, frontier performance | ~$3/M in, ~$15/M out |
| `xiaomi/mimo-v2-pro` | **Default.** Budget pick, 1T MoE, strong agentic tool use | ~$1/M in, ~$3/M out |
| `minimax/minimax-m2.7` | Very cheap, multi-agent planning | ~$0.30/M in, ~$1.20/M out |
| `z-ai/glm-5-turbo` | Fast inference, optimized for agent workflows | ~$1.20/M in, ~$4/M out |

### Ollama Cloud

| Model | Notes |
|-------|-------|
| `minimax-m2.7:cloud` | MiniMax M2.7 via Ollama |
| `glm-5:cloud` | GLM-5 via Ollama |
| `kimi-k2.5:cloud` | Moonshot multimodal agent, free tier available |

> You can specify **any model** available on OpenRouter or Ollama — the table above lists defaults and recommendations.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              auren-forge CLI                │
│  (Typer-based, user commands, YAML config)  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Session Manager                    │
│  (state.json persistent file, decision log,  │
│   crash recovery, session history)           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          Training Monitor                    │
│  (lightweight Python, NO LLM calls)          │
│  - parses Unsloth/HF Trainer callbacks       │
│  - detects anomalies, plateau, NaN, spikes   │
│  - updates state.json                        │
│  - wakes Agent when events trigger           │
└──────────────────┬──────────────────────────┘
                   │ (only on relevant events)
┌──────────────────▼──────────────────────────┐
│           AI Agent (LLM via API)             │
│  - receives: state + metrics + tool defs     │
│  - decides: continue, pause, eval, notify    │
│  - each call is STATELESS (context from      │
│    state.json, not conversation history)     │
│  - uses OpenAI-compatible function calling   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Tools Layer                     │
│  get_training_status    get_loss_history     │
│  pause_training         resume_training      │
│  run_eval               get_model_outputs    │
│  send_notification      validate_dataset     │
│  save_checkpoint        get_session_summary  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          Backend (Unsloth + SFTTrainer)      │
│  (actual model training)                     │
└─────────────────────────────────────────────┘
```

### Memory System

Training runs can last 6–12+ hours. The agent cannot maintain a persistent API session — doing so would exhaust the context window. Instead:

1. **`state.json`** — persistent working memory on disk, updated after every significant event
2. **Metrics buffer** — lightweight in-memory rolling statistics, no LLM involvement
3. **Agent calls** — stateless, on-demand; all context is reconstructed from `state.json` each time

---

## Anomaly Detection

The Training Monitor watches for:

| Anomaly | Threshold |
|---------|-----------|
| NaN loss | Any non-finite loss → immediate pause |
| Inf loss | Any non-finite loss → immediate pause |
| Loss spike | current_loss > 3× rolling avg (50-step window) |
| Loss plateau | Linear regression slope < 1e-6 over last 500 steps |
| Loss divergence | Increasing for 500+ consecutive steps |
| Gradient explosion | grad_norm > 10× rolling avg |

---

## Dataset Formats

### Alpaca
```json
{"instruction": "...", "input": "...", "output": "..."}
```

### ShareGPT
```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```

### Custom
Set `column_mapping` in your config:
```yaml
dataset:
  format: "custom"
  column_mapping:
    instruction: "question"
    output: "answer"
```

---

## Running Tests

```bash
pip install "auren-forge[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by [Auren Research](https://github.com/meryyllea/auren-forge)*
