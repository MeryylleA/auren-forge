# auren-forge

**AI-orchestrated supervised fine-tuning for language models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Auren Research](https://img.shields.io/badge/by-Auren%20Research-b8860b.svg)](https://github.com/meryyllea/auren-forge)

```
   █████╗ ██╗   ██╗██████╗ ███████╗███╗   ██╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔══██╗██║   ██║██╔══██╗██╔════╝████╗  ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ███████║██║   ██║██████╔╝█████╗  ██╔██╗ ██║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██╔══██║██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ██║  ██║╚██████╔╝██║  ██║███████╗██║ ╚████║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

---

## What is auren-forge?

auren-forge is an open-source **interactive TUI application** that automates SFT post-training of language models using an AI agent as orchestrator. You launch it with `forge`, walk through a guided setup, and watch training happen live — while an LLM agent monitors metrics in real-time, detects anomalies, runs evaluations, and notifies you when something needs attention.

It's like `htop` for your training run, with an AI co-pilot built in.

Key ideas:

- **You stay in the loop**: the agent always asks before starting a new training round. Autonomous actions (pause on NaN, run eval) are configurable and scoped.
- **Works unattended**: designed for 6–12+ hour training runs. The agent is called on-demand, not in a persistent API session — no context window exhaustion, no spiraling costs.
- **Crash-safe**: all state lives in `state.json` on disk. Kill the process, restart, resume from the last checkpoint.
- **Universal model and dataset support**: any HuggingFace model supported by Unsloth, any Alpaca/ShareGPT/custom dataset.

---

## TUI Demo

```
╭─── AUREN FORGE — Training: llama3-alpaca — RUNNING ──────────────────────────╮
│                                                                               │
│ ╭─ Metrics ─────────────────────────────────╮  ╭─ Agent Decisions ─────────╮ │
│ │ Step:     1,240 / 12,890  (9.6%)          │  │ [14:23] training_start    │ │
│ │ Epoch:    0.10                            │  │   Training started.       │ │
│ │ Loss:     0.9430                          │  │   Monitoring metrics.     │ │
│ │ Best:     0.8910  @ step 1180             │  │                           │ │
│ │ Trend:    ▼ decreasing                    │  │ [14:35] scheduled_interval│ │
│ │ LR:       1.80e-4                         │  │   Loss decreasing. OK.    │ │
│ │ Grad:     0.420                           │  │                           │ │
│ │ Elapsed:  22.0 min   ETA: 134.0 min       │  │ [14:47] scheduled_interval│ │
│ │ Status:   training                        │  │   Trend healthy.          │ │
│ ╰───────────────────────────────────────────╯  ╰───────────────────────────╯ │
│                                                                               │
│  Loss Curve:                                                                  │
│  1.800│██                                                                     │
│  1.575│ ██                                                                    │
│  1.350│   ██                                                                  │
│  1.125│     ████                                                              │
│  0.900│         ██████████████████▁▁▁▁▁▁▁▁▁                                 │
│  0.675│                                     ▁▁▁▁▁▁▁▁▁                       │
│        └────────────────────────────────────────────────────                 │
│        latest: 0.9430  best: 0.8910  points: 124                             │
│                                                                               │
│  [P] Pause  [E] Eval  [C] Chat  [Esc] Home                                   │
╰───────────────────────────────────────────────────────────────────────────────╯
```

---

## Installation

```bash
# Base install (TUI + config, no GPU required)
pip install auren-forge

# Full install with training dependencies (requires CUDA)
pip install "auren-forge[train]"

# From source
git clone https://github.com/meryyllea/auren-forge
cd auren-forge
pip install -e ".[train,dev]"
```

---

## Quick Start

```bash
# Launch the TUI
forge
```

That's it. From there:

1. Press **N** or click **New Project** — walks you through a 4-step setup wizard
2. Configure model, dataset, training hyperparameters, and AI agent provider
3. Press **🚀 Start Training** — training launches in a background thread
4. Watch metrics update live on the Training Monitor screen
5. Press **C** to chat with the agent, **E** to trigger eval, **P** to pause/resume
6. When training completes, the agent scores model outputs and reports quality
7. If score is below threshold, the agent recommends data to improve — you confirm

---

## Screens

| Screen | Description |
|--------|-------------|
| **Home** | Landing page with recent sessions list |
| **Setup Wizard** | 4-step guided config: model → dataset → training → agent |
| **Training Monitor** | Live metrics, loss curve, agent decisions log |
| **Agent Chat** | Conversational interface with the Forge agent |
| **Eval Results** | Evaluation scores, agent assessment, option to start round 2 |
| **History** | Browse all past sessions, resume or export |
| **Settings** | Global defaults saved to `~/.config/auren-forge/settings.yaml` |

Keyboard shortcuts work throughout: **N/R/H/S** on home, **P/E/C** during training, **Esc** goes back.

---

## Features

- **Live TUI** — full-screen terminal application (Textual), not a CLI. Panels update every 2 seconds during training.
- **AI agent orchestrator** — LLM monitors training, detects anomalies, runs evaluations, suggests fixes. Configurable autonomy: `monitor` / `suggest` / `auto`.
- **Loss chart** — ASCII sparkline rendered in the terminal, updates every logging step.
- **Anomaly detection** — NaN/Inf loss, spikes (3× rolling avg), plateaus (linear regression), divergence (500+ steps increasing), gradient explosions. Agent is woken immediately on critical events.
- **Crash recovery** — `state.json` on disk with atomic writes. `forge` → Resume Session picks up from the last checkpoint.
- **Multi-round SFT** — if eval score falls below threshold, agent recommends a new dataset. You confirm, training restarts. All rounds tracked in session history.
- **Dataset auto-detection** — Alpaca, ShareGPT, or custom column mapping. Validation reports empty rows, format issues, and size warnings before training starts.
- **Model export** — safetensors, HuggingFace format, or GGUF.

---

## Supported Models (Training)

Any HuggingFace model supported by Unsloth. Popular 4-bit options:

| Model | Size | Notes |
|-------|------|-------|
| `unsloth/Llama-3.1-8B-bnb-4bit` | 8B | Default recommendation |
| `unsloth/Llama-3.2-3B-bnb-4bit` | 3B | Fast, low VRAM |
| `unsloth/mistral-7b-v0.3-bnb-4bit` | 7B | Strong base model |
| `unsloth/Phi-3.5-mini-instruct-bnb-4bit` | 3.8B | Excellent for instruction tuning |
| `unsloth/Qwen2.5-7B-bnb-4bit` | 7B | Strong multilingual |
| `unsloth/gemma-2-9b-bnb-4bit` | 9B | Google's Gemma 2 |

---

## Agent Providers

### OpenRouter

| Model | Notes | Cost |
|-------|-------|------|
| `anthropic/claude-opus-4.6` | **Primary.** Best judgment, 1M context | ~$5/$25 per M tokens |
| `anthropic/claude-sonnet-4.6` | Best cost/capability balance | ~$3/$15 per M tokens |
| `xiaomi/mimo-v2-pro` | **Default.** 1T MoE, strong tool use | ~$1/$3 per M tokens |
| `minimax/minimax-m2.7` | Very cheap, multi-agent planning | ~$0.30/$1.20 per M tokens |
| `z-ai/glm-5-turbo` | Fast, optimized for agent workflows | ~$1.20/$4 per M tokens |

### Ollama Cloud

| Model | Notes |
|-------|-------|
| `minimax-m2.7:cloud` | MiniMax M2.7 via Ollama |
| `glm-5:cloud` | GLM-5 via Ollama |
| `kimi-k2.5:cloud` | Moonshot multimodal agent, free tier |

Set `OPENROUTER_API_KEY` in your environment, or enter the key in the Setup Wizard or Settings screen.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           auren-forge TUI (Textual)          │
│  Screens: Home, Setup, Training, Chat,       │
│  EvalResults, History, Settings              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Session Manager                    │
│  state.json — atomic writes, crash recovery  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          Training Monitor (no LLM)           │
│  - HF TrainerCallback → rolling stats        │
│  - Anomaly detection: NaN, spike, plateau,   │
│    divergence, grad explosion                │
│  - Wakes agent on events (debounced)         │
└──────────────────┬──────────────────────────┘
                   │ on relevant events only
┌──────────────────▼──────────────────────────┐
│           AI Agent (stateless LLM calls)     │
│  - Full context rebuilt from state.json      │
│  - Multi-round tool calling                  │
│  - Decisions logged with reasoning           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Tools (10 functions)            │
│  get_training_status  get_loss_history       │
│  pause_training       resume_training        │
│  run_eval             get_model_outputs      │
│  send_notification    validate_dataset       │
│  save_checkpoint      get_session_summary    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          Backend: Unsloth + SFTTrainer       │
│  (runs in background thread)                 │
└─────────────────────────────────────────────┘
```

---

## How It Works

**A typical session:**

1. **Launch** `forge` — the TUI opens on the Home screen
2. **Setup Wizard** — enter model ID, dataset, hyperparameters, and API key in 4 steps. Inline dataset validation tells you row count and detected format before you commit.
3. **Training starts** in a background thread. The TUI remains responsive.
4. **Every ~30 minutes** (configurable), the agent wakes up, reviews the current `state.json`, and decides: continue, flag an anomaly, or notify you.
5. **Anomalies** (NaN, loss spike >3×, divergence) wake the agent immediately. In `auto` mode it pauses training; in `suggest` mode it notifies you and waits.
6. **On completion**, the agent generates 15 diverse test prompts, runs them through the model, and scores the outputs on relevance, coherence, instruction-following, and factual accuracy (1–5 each).
7. **If the score is below threshold** (default 3.0/5.0), the Eval Results screen shows what's weak and suggests datasets to fix it. You decide whether to run another round.

---

## Why the Agent Doesn't Hallucinate

Each agent call is **stateless** — there is no persistent conversation history between calls. Every call reconstructs full context from `state.json`:

```
trigger + state.json + last 50 metrics → LLM → tool calls → logged decision
```

This means:
- No context drift over a 12-hour run
- No stale information from earlier in the conversation
- Cost is proportional to events, not to training duration
- A crashed and resumed session gives the agent identical context

---

## Memory System

| Layer | What it is | Who updates it |
|-------|-----------|----------------|
| `state.json` | Persistent JSON on disk | Session Manager (atomic writes) |
| Metrics buffer | In-memory rolling stats | Training Monitor (no LLM) |
| Agent calls | On-demand, stateless | Triggered by Monitor events |

---

## Configuration

`forge.yaml` (generated by the Setup Wizard, or copy from `configs/default.yaml`):

```yaml
project_name: "my-sft-project"

model:
  name: "unsloth/Llama-3.1-8B-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

dataset:
  source: "yahma/alpaca-cleaned"
  format: "auto"   # auto | alpaca | sharegpt | custom
  split: "train"

training:
  lora_rank: 16
  lora_alpha: 16
  batch_size: 2
  gradient_accumulation_steps: 4
  epochs: 1
  learning_rate: 2.0e-4
  output_dir: "./output"

agent:
  provider: "openrouter"
  api_key: "${OPENROUTER_API_KEY}"
  model: "xiaomi/mimo-v2-pro"
  autonomy: "suggest"           # monitor | suggest | auto
  check_interval_minutes: 30
  eval_on_completion: true

eval:
  num_test_prompts: 15
  min_quality_score: 3.0
```

Global defaults (provider, API key, etc.) are also editable in the **Settings** screen and saved to `~/.config/auren-forge/settings.yaml`.

---

## Development

```bash
git clone https://github.com/meryyllea/auren-forge
cd auren-forge
pip install -e ".[dev]"

# Run tests (no GPU required)
pytest tests/

# Launch the TUI
python -m forge
```

Tests cover config validation, session CRUD, anomaly detection, dataset format detection, and agent tool dispatch — 73 tests, all runnable without GPU.

---

## Dataset Formats

| Format | Required columns | Example dataset |
|--------|-----------------|-----------------|
| Alpaca | `instruction`, `output` (+ optional `input`) | `yahma/alpaca-cleaned` |
| ShareGPT | `conversations` (list of `{from, value}`) | `anon8231489123/ShareGPT_Vicuna_unfiltered` |
| Custom | Any — set `column_mapping` in config | Your own data |

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by [Auren Research](https://github.com/meryyllea/auren-forge)*
