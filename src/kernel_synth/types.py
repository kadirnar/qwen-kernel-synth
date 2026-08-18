from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_BACKENDS = frozenset({"triton", "cuda"})


@dataclass(frozen=True, slots=True)
class SeedTask:
    """A deterministic reference task that a teacher model must optimize."""

    id: str
    operation: str
    objective: str
    reference_code: str
    backend: str = "triton"
    constraints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> SeedTask:
        required = ("id", "operation", "objective", "reference_code")
        missing = [key for key in required if not value.get(key)]
        if missing:
            raise ValueError(f"seed is missing required fields: {', '.join(missing)}")

        backend = str(value.get("backend", "triton")).lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"unsupported backend {backend!r}; choose one of {sorted(SUPPORTED_BACKENDS)}"
            )

        constraints = value.get("constraints", [])
        if not isinstance(constraints, list) or not all(
            isinstance(item, str) for item in constraints
        ):
            raise ValueError("seed constraints must be a list of strings")

        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("seed metadata must be an object")

        return cls(
            id=str(value["id"]),
            operation=str(value["operation"]),
            objective=str(value["objective"]),
            reference_code=str(value["reference_code"]),
            backend=backend,
            constraints=tuple(constraints),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Normalized fields returned by OpenRouter."""

    content: str
    response_id: str
    requested_model: str
    actual_model: str
    provider: str | None
    usage: dict[str, Any]
    raw_response: dict[str, Any]

    @property
    def cost(self) -> float:
        value = self.usage.get("cost", 0.0)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
