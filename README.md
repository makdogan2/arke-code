# Arke Code

A local, privacy-first coding assistant that runs entirely on your own GPU.
Arke Code is built on **Qwen3-Coder 30B-A3B**, served through **Ollama**, with a custom
mentor-style persona and a roadmap toward tool use, project memory and eval-gated fine-tuning.

The name comes from the Greek *arkhe*: "first principle", the starting point of everything.

## Features

- Runs 100% locally: no API keys, no data leaves your machine
- Streaming responses with conversation memory
- Chat commands: `/paste`, `/long`, `/short`, `/temp`, `/clear`, `/settings`, `/quit`
- Warm, concise persona defined in the `Modelfile`

## Hardware

Developed on an RTX 5070 Ti (16 GB VRAM) with 32 GB RAM.
The base model is a Mixture-of-Experts (30B total, ~3B active per token, ~18 GB),
so Ollama splits it between VRAM and system RAM and it still runs fast.

## Quick start

1. Install [Ollama](https://ollama.com)
2. `ollama pull qwen3-coder`
3. `ollama create arke-code -f Modelfile`
4. `pip install -r requirements.txt`
5. `python assistant.py`

## Project structure

| File | Purpose |
|------|---------|
| `assistant.py` | CLI chat client using the Ollama REST API |
| `Modelfile` | Base model, system prompt and parameters |
| `download_datasets.py` | Builds a combined instruction dataset (CodeAlpaca, Evol-Instruct-Code, CodeFeedback) |
| `training_data.py` | Small hand-written dataset from early experiments |
| `finetune.py`, `finetune_14b.py` | QLoRA fine-tuning scripts (4-bit, LoRA r=16, gradient checkpointing) |
| `web_chat.py` | Gradio web UI (legacy, from the transformers-based version) |

## Roadmap

- [x] **Phase 0:** Ollama-based assistant with a custom persona
- [ ] **Phase 1:** Evaluation suite (custom tasks + HumanEval subset)
- [ ] **Phase 2:** Agentic tools (read/write files, run tests, git diff) with confirmation
- [ ] **Phase 3:** Project memory (RAG with local embeddings)
- [ ] **Phase 4:** VS Code integration
- [ ] **Phase 5:** Continuous, eval-gated fine-tuning (QLoRA, GGUF export, back into Ollama)

## History

The project started as **Icarus**, a series of experiments fine-tuning Qwen models with LoRA/QLoRA
on a single 16 GB GPU. Along the way it hit CUDA out-of-memory errors, device mismatches and
100 MB GitHub file limits, and each fix shaped the current design: serve a strong base model
locally, measure everything, and only keep fine-tunes that beat the baseline.
