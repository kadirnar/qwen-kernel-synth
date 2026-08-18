from __future__ import annotations

import hashlib
import json

from .types import SeedTask

SYSTEM_PROMPT = """You are a senior GPU kernel engineer creating supervised training data.
Produce a correct, self-contained replacement for the supplied PyTorch reference program.

Hard requirements:
- Preserve the reference program's mathematical semantics, shapes, dtypes, and public helpers.
- Implement real device work with the requested backend. Do not merely define an unused kernel.
- Do not call torch.compile or the end-to-end PyTorch operation being replaced.
- Do not return constants, cache answers for known inputs, skip required operations, or add a
  dead/ghost optimization branch.
- Do not access the network, filesystem, shell, environment variables, or external services.
- Keep bounds checks and accumulate reductions in a numerically appropriate dtype.
- Return a complete Python program containing ModelNew, get_inputs, and get_init_inputs.
- Describe only a concise engineering rationale. Never expose private chain-of-thought.
- Output only an object matching the provided JSON schema.

The examples are original synthetic tasks. Do not reproduce or quote KernelBench evaluation tasks.
"""


MUTATION_PROFILES = (
    "prioritize coalesced memory access and minimize intermediate tensors",
    "look for safe fusion opportunities and reduce kernel-launch overhead",
    "use shape-aware tiling while retaining correct tail masks",
    "balance occupancy, register pressure, and arithmetic intensity",
    "use numerically stable reductions with explicit accumulation precision",
    "specialize only on invariants guaranteed by get_inputs and get_init_inputs",
)


KERNEL_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthetic_kernel_candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "instruction": {"type": "string", "minLength": 20},
                "backend": {"type": "string", "enum": ["triton", "cuda"]},
                "reasoning_summary": {"type": "string", "minLength": 20},
                "estimated_bottleneck": {"type": "string", "minLength": 3},
                "optimization_notes": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 8,
                    "items": {"type": "string", "minLength": 3},
                },
                "assumptions": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
                "test_strategy": {"type": "string", "minLength": 20},
                "solution_code": {"type": "string", "minLength": 100},
            },
            "required": [
                "instruction",
                "backend",
                "reasoning_summary",
                "estimated_bottleneck",
                "optimization_notes",
                "assumptions",
                "test_strategy",
                "solution_code",
            ],
        },
    },
}


def mutation_profile(seed_id: str, sample_index: int) -> str:
    digest = hashlib.sha256(f"{seed_id}:{sample_index}".encode()).digest()
    return MUTATION_PROFILES[int.from_bytes(digest[:2], "big") % len(MUTATION_PROFILES)]


def build_messages(seed: SeedTask, sample_index: int) -> list[dict[str, str]]:
    constraints = "\n".join(f"- {item}" for item in seed.constraints) or "- None beyond code"
    user_prompt = f"""Create synthetic GPU-kernel training example {sample_index + 1} for this seed.

Operation: {seed.operation}
Objective: {seed.objective}
Required backend: {seed.backend}
Variation focus: {mutation_profile(seed.id, sample_index)}

Additional constraints:
{constraints}

Reference program:
```python
{seed.reference_code.strip()}
```

The solution must be independent of known benchmark answers. Explain the proposed optimization in
reasoning_summary, but put the complete executable replacement only in solution_code.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def schema_json() -> str:
    return json.dumps(KERNEL_RESPONSE_SCHEMA, separators=(",", ":"), sort_keys=True)
