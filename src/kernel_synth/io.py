from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import SeedTask


def load_seeds(path: Path) -> list[SeedTask]:
    seeds: list[SeedTask] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            try:
                seed = SeedTask.from_mapping(value)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if seed.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate seed id {seed.id!r}")
            seen.add(seed.id)
            seeds.append(seed)
    if not seeds:
        raise ValueError(f"{path} contains no seed tasks")
    return seeds


def completed_job_keys(*paths: Path) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("job_key")
                if isinstance(key, str):
                    keys.add(key)
    return keys


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows
