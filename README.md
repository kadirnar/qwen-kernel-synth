# qwen-kernel-synth

Generate resumable, auditable CUDA and Triton SFT records through OpenRouter. The project is
designed for building a kernel-specialized Qwen training corpus without silently mixing raw model
output with verified data.

The default model is `z-ai/glm-5.2`, with Kimi K2.7 Code and DeepSeek V4 Flash as request
fallbacks. See [the model research](docs/model-selection.md) for the quality/cost analysis.

## What the pipeline does

- Reads original PyTorch reference tasks from JSONL.
- Requests strict JSON Schema output from OpenRouter.
- Uses deterministic variation profiles to create multiple candidates per task.
- Retries transient errors and honors `Retry-After`.
- Denies provider data collection by default and optionally requires ZDR endpoints.
- Statically rejects malformed, unsafe, non-kernel, and reward-hacking-prone responses.
- Writes raw, accepted, and rejected append-only JSONL files.
- Records the actual routed model, provider, response ID, token usage, and cost.
- Resumes without regenerating completed jobs.

The budget guard stops scheduling new requests after OpenRouter-reported spend reaches the limit.
Already in-flight requests can make the final cost exceed that limit slightly; lower `--concurrency`
when the cap must be tight.

Static acceptance does **not** mean that a kernel is correct or fast. Accepted rows deliberately
contain `"gpu_validation":"not_run"` until an isolated GPU verifier updates them.

## Installation

```bash
uv sync --extra dev
cp .env.example .env
```

Export the API key in your shell; the CLI intentionally does not read or store `.env` files:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Show live availability and pricing for the recommended models:

```bash
uv run kernel-synth models
```

## Quick start

Generate six candidates from the three included example seeds:

```bash
uv run kernel-synth generate \
  --seeds data/seeds.example.jsonl \
  --target-rows 6 \
  --budget-usd 5
```

## Generate data

Start with the original, CC0 seed examples:

```bash
uv run kernel-synth generate \
  --seeds data/seeds.example.jsonl \
  --samples-per-seed 2 \
  --concurrency 4 \
  --budget-usd 5
```

`--target-rows` sets an exact number of candidate requests and distributes them evenly across all
seeds. The generator uses a bounded worker queue, so a large target does not create every async task
in memory. A million-row run still needs many diverse seeds; repeatedly sampling the three examples
will create low-value duplicates.

Outputs are written under `data/generated/`:

- `raw.jsonl`: exact normalized model content and OpenRouter metadata.
- `accepted.jsonl`: static-check-passing SFT records.
- `rejected.jsonl`: API failures, invalid JSON, unsafe code, and validation failures.

Use a premium teacher for a small gold-generation or review pass:

```bash
uv run kernel-synth generate \
  --seeds data/seeds.example.jsonl \
  --model anthropic/claude-opus-5 \
  --fallback z-ai/glm-5.2 \
  --samples-per-seed 1 \
  --reasoning-effort high \
  --budget-usd 20
```

Use a low-cost draft pass:

```bash
uv run kernel-synth generate \
  --seeds data/seeds.example.jsonl \
  --model deepseek/deepseek-v4-flash-0731 \
  --fallback qwen/qwen3-coder \
  --samples-per-seed 4 \
  --temperature 0.65
```

Re-run static validation later:

```bash
uv run kernel-synth validate --input data/generated/accepted.jsonl
```

## Seed format

Each line of the seed file is one object:

```json
{
  "id": "unique-original-task-id",
  "operation": "short operation family",
  "objective": "what should be optimized",
  "backend": "triton",
  "constraints": ["shape and numerical requirements"],
  "reference_code": "complete PyTorch Model/get_inputs/get_init_inputs program",
  "metadata": {"license": "CC0-1.0", "origin": "original"}
}
```

Only use references that you created or are licensed for training and redistribution. Keep
evaluation suites such as KernelBench in a separate repository or explicit denylist.

## Privacy and routing

By default, requests include:

```json
{
  "provider": {
    "data_collection": "deny",
    "require_parameters": true,
    "allow_fallbacks": true
  }
}
```

Add `--zdr` to require a Zero Data Retention endpoint. This can reduce model/provider availability.
Use `--allow-provider-data-collection` only when the seed code is safe to share with such providers.

## Required GPU validation stage

Do not train directly on `accepted.jsonl`. Run generated code inside a disposable, network-disabled
GPU container with resource and time limits. A proper verifier should record:

- compilation status and compiler diagnostics;
- randomized numerical comparisons over multiple seeds and edge shapes;
- maximum absolute and relative error;
- evidence that the custom kernel actually executes;
- device, CUDA/ROCm, PyTorch, Triton, driver, and compiler versions;
- warmup count, repetitions, median/p5/p95 latency, and baseline speedup;
- static and dynamic reward-hacking checks.

Only rows passing that stage should become `verified` SFT data. Failed candidates remain useful for
verifier, repair, and preference training.

## Development

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```
