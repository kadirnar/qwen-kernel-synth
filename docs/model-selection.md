# OpenRouter teacher-model selection

Research snapshot: 2026-08-18. Prices and availability change; run
`kernel-synth models` before a large job to read the live OpenRouter catalog.

## Recommendation

Use `z-ai/glm-5.2` as the default practical teacher, with
`moonshotai/kimi-k2.7-code` and `deepseek/deepseek-v4-flash-0731` as fallbacks.
For a smaller, carefully curated gold set, generate or review the same seeds with
`anthropic/claude-opus-5`.

This recommendation separates two needs:

1. High-volume diversity at a sustainable price.
2. A smaller high-quality set reviewed by a model that performs well on real kernel-agent tasks.

Do not select a teacher from generic coding benchmarks alone. GPU kernels require numerical
correctness, hardware-aware scheduling, profiling feedback, and resistance to benchmark reward
hacks. The generated code still needs compilation, randomized correctness tests, and GPU timing.

## Candidate comparison

| Model | Role | OpenRouter list price at research time | Why |
|---|---|---:|---|
| `anthropic/claude-opus-5` | Premium teacher/reviewer | $5 input / $25 output per 1M tokens | Strongest overall choice visible on current KernelBench CUDA results, but expensive for bulk synthesis. |
| `z-ai/glm-5.2` | Recommended practical teacher | $0.50 / $3.15 per 1M in the API snapshot | Long-context reasoning/coding model. A public KernelGym release contains 3,200 correctness-passing GLM-5.2 CUDA/Triton trajectories, making it unusually relevant to this task. |
| `moonshotai/kimi-k2.7-code` | Code-focused bulk teacher | $0.71 / $3.50 per 1M in the API snapshot | Code-specialized, supports reasoning and structured output, and has substantially lower cost than premium teachers. |
| `deepseek/deepseek-v4-flash-0731` | Cheap diversification | $0.14 / $0.28 per 1M | Very inexpensive coding/reasoning model; useful for candidate diversity and some hardware-specialized tasks, but not a substitute for execution validation. |
| `qwen/qwen3-coder` | Open-weight baseline | $0.30 / $1.00 per 1M in the API snapshot | Strong code model and useful for self-distillation comparisons. Avoid making it the only teacher for a Qwen student, because teacher diversity is valuable. |
| `qwen/qwen3-coder-30b-a3b-instruct` | Cheapest Qwen baseline | $0.07 / $0.28 per 1M in the API snapshot | Good for prompt and pipeline experiments. Too weak to trust as the only source of gold kernel solutions. |

## Evidence and API sources

- Kernel-specialized comparison: <https://kernelbench.com/>
- OpenRouter model catalog API: <https://openrouter.ai/api/v1/models>
- Claude Opus 5: <https://openrouter.ai/anthropic/claude-opus-5>
- GLM 5.2: <https://openrouter.ai/z-ai/glm-5.2>
- Kimi K2.7 Code: <https://openrouter.ai/moonshotai/kimi-k2.7-code>
- DeepSeek V4 Flash 0731: <https://openrouter.ai/deepseek/deepseek-v4-flash-0731>
- Qwen3 Coder: <https://openrouter.ai/qwen/qwen3-coder>
- Structured outputs: <https://openrouter.ai/docs/guides/features/structured-outputs>
- Reasoning controls: <https://openrouter.ai/docs/guides/best-practices/reasoning-tokens>
- Provider routing and privacy: <https://openrouter.ai/docs/guides/routing/provider-selection>
- Usage and cost accounting: <https://openrouter.ai/docs/cookbook/administration/usage-accounting>

## Production generation strategy

For each original seed task:

1. Generate two to four candidates using different mutation profiles and fixed request seeds.
2. Require JSON Schema output so malformed records do not silently enter the dataset.
3. Apply static checks and quarantine rejected responses.
4. Compile in an isolated GPU worker. Never execute model-generated code on the data-orchestration
   host.
5. Compare `ModelNew` to the reference across randomized shapes, values, dtypes, and edge cases.
6. Benchmark only after correctness passes. Store GPU model, software versions, warmup, repeats,
   median latency, and dispersion.
7. Keep only genuinely custom kernels. Reject `torch.compile`, end-to-end framework fallbacks,
   constants, dead kernel branches, and unused kernels.
8. Deduplicate reference tasks and normalized solutions before splitting by task family.
9. Keep KernelBench and other evaluation tasks in a denylist and out of training.

The pipeline in this repository implements steps 1–3 and records `gpu_validation: not_run` so
unverified model output cannot be confused with benchmarked gold data.
