from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .types import SeedTask

REQUIRED_FIELDS = (
    "instruction",
    "backend",
    "reasoning_summary",
    "estimated_bottleneck",
    "optimization_notes",
    "assumptions",
    "test_strategy",
    "solution_code",
)

UNSAFE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bsubprocess\b", "subprocess use"),
    (r"\bos\.system\s*\(", "shell execution"),
    (r"\bos\.popen\s*\(", "shell execution"),
    (r"\b(?:requests|urllib|socket)\b", "network access"),
    (r"\bopen\s*\(", "filesystem access"),
    (r"\b(?:eval|exec)\s*\(", "dynamic code execution"),
    (r"torch\.compile\s*\(", "torch.compile fallback"),
)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    accepted: bool
    issues: tuple[str, ...]
    solution_hash: str


def parse_candidate(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model output must be a JSON object")
    return value


def validate_candidate(candidate: dict[str, Any], seed: SeedTask) -> ValidationResult:
    issues: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in candidate]
    if missing:
        issues.append(f"missing fields: {', '.join(missing)}")

    backend = str(candidate.get("backend", "")).lower()
    if backend != seed.backend:
        issues.append(f"backend {backend!r} does not match required {seed.backend!r}")

    solution = str(candidate.get("solution_code", ""))
    if "```" in solution:
        issues.append("solution_code contains Markdown fences")
    if len(solution) < 100:
        issues.append("solution_code is too short")

    for symbol in ("class ModelNew", "def get_inputs", "def get_init_inputs"):
        if symbol not in solution:
            issues.append(f"solution_code is missing {symbol}")

    if seed.backend == "triton" and not all(
        marker in solution for marker in ("import triton", "@triton.jit")
    ):
        issues.append("Triton solution does not define a @triton.jit kernel")
    if seed.backend == "cuda" and "__global__" not in solution:
        issues.append("CUDA solution does not define a __global__ kernel")

    for pattern, description in UNSAFE_PATTERNS:
        if re.search(pattern, solution):
            issues.append(f"solution contains disallowed {description}")

    try:
        ast.parse(solution)
    except SyntaxError as exc:
        issues.append(f"Python syntax error at line {exc.lineno}: {exc.msg}")

    notes = candidate.get("optimization_notes", [])
    assumptions = candidate.get("assumptions", [])
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        issues.append("optimization_notes must be a string list")
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        issues.append("assumptions must be a string list")

    solution_hash = hashlib.sha256(_normalize_code(solution).encode()).hexdigest()
    return ValidationResult(not issues, tuple(issues), solution_hash)


def render_assistant_text(candidate: dict[str, Any]) -> str:
    notes = "\n".join(f"- {item}" for item in candidate["optimization_notes"])
    assumptions = "\n".join(f"- {item}" for item in candidate["assumptions"]) or "- None"
    return (
        f"Approach: {candidate['reasoning_summary']}\n\n"
        f"Estimated bottleneck: {candidate['estimated_bottleneck']}\n\n"
        f"Optimization notes:\n{notes}\n\n"
        f"Assumptions:\n{assumptions}\n\n"
        f"Test strategy: {candidate['test_strategy']}\n\n"
        f"```python\n{candidate['solution_code'].rstrip()}\n```"
    )


def stable_record_id(seed_id: str, sample_index: int, solution_hash: str) -> str:
    raw = f"{seed_id}:{sample_index}:{solution_hash}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _normalize_code(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())
