from __future__ import annotations

from typing import Any

import httpx

RECOMMENDED_MODELS: tuple[tuple[str, str], ...] = (
    ("anthropic/claude-opus-5", "premium teacher / final review"),
    ("z-ai/glm-5.2", "recommended practical teacher"),
    ("moonshotai/kimi-k2.7-code", "code-focused bulk teacher"),
    ("deepseek/deepseek-v4-flash-0731", "low-cost diverse drafts"),
    ("qwen/qwen3-coder", "open-weight baseline / distillation"),
    ("qwen/qwen3-coder-30b-a3b-instruct", "very low-cost baseline"),
)


def fetch_recommended_models(timeout: float = 30.0) -> list[dict[str, Any]]:
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=timeout)
    response.raise_for_status()
    available = {row["id"]: row for row in response.json()["data"]}
    rows: list[dict[str, Any]] = []
    for model_id, role in RECOMMENDED_MODELS:
        if model := available.get(model_id):
            rows.append(
                {
                    "id": model_id,
                    "role": role,
                    "context_length": model.get("context_length"),
                    "input_per_million": _per_million(model.get("pricing", {}).get("prompt")),
                    "output_per_million": _per_million(model.get("pricing", {}).get("completion")),
                    "structured_outputs": "structured_outputs"
                    in (model.get("supported_parameters") or []),
                }
            )
    return rows


def _per_million(value: Any) -> float | None:
    try:
        return round(float(value) * 1_000_000, 6)
    except (TypeError, ValueError):
        return None
