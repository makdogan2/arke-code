# Arke Code

A local, privacy-first coding assistant that runs entirely on your own GPU.
Arke Code is built on **Qwen3-Coder 30B-A3B**, served through **Ollama**, with a custom
mentor-style persona and a roadmap toward tool use, project memory and eval-gated fine-tuning.

The name comes from the Greek *arkhe*: "first principle", the starting point of everything.

## Features

- Runs 100% locally: no API keys, no data leaves your machine
- Streaming responses with conversation memory
- Works on your project through Ollama tool calling, confined to the project folder:
  - reads freely: `list_dir`, `read_file`, `grep`, `git_diff`
  - asks first, showing a diff or the exact command: `edit_file`, `write_file`, `run_python`,
    `run_tests` (default answer is no)
- Chat commands: `/paste`, `/long`, `/short`, `/temp`, `/clear`, `/settings`, `/quit`
- Warm, concise persona defined in the `Modelfile`

## Results (v0.1)

Measured with `eval.py` on an RTX 5070 Ti, temperature 0, one attempt per task (pass@1).

| Benchmark | Score |
|-----------|-------|
| HumanEval, all 164 problems | **149 / 164 (90.9%)** |
| Custom suite, 22 tasks | **20 / 22 (90.9%)** |
| Generation speed | ~90 tokens/s |

HumanEval is scored by this repo's own harness (the model returns the full function, which
is run against the official tests), not the official evaluation script.

What the numbers taught so far:

- **Known weaknesses:** accepts out-of-order input such as `"1m1h"` when asked to reject it, and
  writes `assert x == True` in tests even when told not to. These are the first targets for
  fine-tuning.
- **Prompt rules vs. training:** adding explicit engineering rules to the system prompt scored
  20/22 on the custom suite, the same as the plain prompt. Habits like `== True` did not
  change, which is the case for fine-tuning rather than more prompting.
- **Noise:** repeated runs of the same model differ by about one task, so a change has to win
  by more than that, across several runs, before it counts as an improvement.
- **Measure the measurement:** an early version of the harness told test-writing tasks to
  return "the complete code", which made the model paste the function under test. Fixing the
  instruction, not the model, moved that task from fail to pass.

## Try the agent

`examples/buggy_stats` is a tiny project with one failing test:

```
python assistant.py --root examples/buggy_stats
You > The tests are failing. Find the bug and fix it.
```

Arke Code runs the tests, reads the code, proposes an edit as a diff, and runs the tests
again once you approve.

## Hardware

Developed on an RTX 5070 Ti (16 GB VRAM) with 32 GB RAM.
The base model is a Mixture-of-Experts (30B total, ~3B active per token, ~18 GB),
so Ollama splits it between VRAM and system RAM and it still runs fast.

## Quick start

1. Install [Ollama](https://ollama.com)
2. `ollama pull qwen3-coder`
3. `ollama create arke-code -f Modelfile`
4. `pip install -r requirements.txt`
5. `python assistant.py` (or `python assistant.py --root path/to/your/project`)

## Evaluation

`eval.py` measures the model with numbers instead of impressions. Every training round
must beat the last recorded score before it replaces the current model.

```
python eval.py --check                        # verify the eval set itself (no model needed)
python eval.py                                # 22 custom tasks on arke-code
python eval.py --suite humaneval --limit 40   # first 40 HumanEval problems
python eval.py --model qwen3-coder --suite all
```

- **Custom suite** (`evals/custom_tasks.py`): 20 implementation and bug-fix tasks checked by
  hidden tests, plus 2 test-writing tasks. Written tests must pass on the correct function,
  catch every planted bug (mutation testing) and avoid `== True` comparisons.
- **HumanEval** ([OpenAI](https://github.com/openai/human-eval), MIT): the standard Python
  function-completion benchmark, cached in `evals/`.
- Generated code runs in a separate process with a timeout. Each run is saved under
  `results/`, and one summary line is appended to `results/scores.csv`.

## Project structure

| File | Purpose |
|------|---------|
| `assistant.py` | CLI chat client using the Ollama REST API, with a tool-calling loop |
| `tools.py` | Project tools confined to the workspace; changes and runs need confirmation |
| `examples/buggy_stats/` | Small project with a planted bug, for trying the agent |
| `eval.py`, `evals/` | Evaluation harness, custom tasks and cached HumanEval data |
| `Modelfile` | Base model, system prompt and parameters |
| `training_data.py`, `training_data.json` | Small hand-written dataset from early experiments |
| `finetune.py` | QLoRA fine-tuning script from early experiments (4-bit, LoRA r=16, gradient checkpointing) |

## Roadmap

- [x] **Phase 0:** Ollama-based assistant with a custom persona
- [x] **Phase 1:** Evaluation suite (custom tasks + HumanEval) with a recorded baseline
- [ ] **Phase 2:** Agentic tools: read, search, edit, run tests and git diff with confirmation done;
  next: an agent eval that scores bug fixing end to end
- [ ] **Phase 3:** Project memory (RAG with local embeddings)
- [ ] **Phase 4:** VS Code integration
- [ ] **Phase 5:** Continuous, eval-gated fine-tuning (QLoRA, GGUF export, back into Ollama)

## History

The project started as **Icarus**, a series of experiments fine-tuning Qwen models with LoRA/QLoRA
on a single 16 GB GPU. Along the way it hit CUDA out-of-memory errors, device mismatches and
100 MB GitHub file limits, and each fix shaped the current design: serve a strong base model
locally, measure everything, and only keep fine-tunes that beat the baseline.
