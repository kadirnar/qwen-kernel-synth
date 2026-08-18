from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .client import AsyncOpenRouterClient, OpenRouterError
from .io import append_jsonl, completed_job_keys
from .prompts import KERNEL_RESPONSE_SCHEMA, build_messages
from .types import SeedTask
from .validation import (
    parse_candidate,
    render_assistant_text,
    stable_record_id,
    validate_candidate,
)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    model: str
    fallbacks: tuple[str, ...]
    output: Path
    rejected_output: Path
    raw_output: Path
    samples_per_seed: int = 1
    target_rows: int | None = None
    concurrency: int = 4
    max_tokens: int = 8_192
    temperature: float = 0.45
    top_p: float = 0.95
    reasoning_effort: str | None = "high"
    budget_usd: float | None = None
    data_collection: str = "deny"
    zdr: bool = False


@dataclass(slots=True)
class GenerationStats:
    requested: int = 0
    skipped: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    cost_usd: float = 0.0


class DatasetGenerator:
    def __init__(
        self,
        *,
        client: AsyncOpenRouterClient,
        config: GenerationConfig,
    ) -> None:
        self.client = client
        self.config = config
        self.stats = GenerationStats()
        self._write_lock = asyncio.Lock()
        self._budget_lock = asyncio.Lock()
        self._stop_scheduling = asyncio.Event()
        self._seen_hashes: set[str] = set()

    async def run(self, seeds: list[SeedTask]) -> GenerationStats:
        completed = completed_job_keys(
            self.config.output,
            self.config.rejected_output,
        )
        queue: asyncio.Queue[tuple[SeedTask, int] | None] = asyncio.Queue(
            maxsize=self.config.concurrency * 2
        )

        async def produce() -> None:
            for seed, sample_index in generation_jobs(
                seeds,
                samples_per_seed=self.config.samples_per_seed,
                target_rows=self.config.target_rows,
            ):
                if self._stop_scheduling.is_set():
                    break
                key = job_key(seed, sample_index, self.config.model)
                if key in completed:
                    self.stats.skipped += 1
                else:
                    self.stats.requested += 1
                    await queue.put((seed, sample_index))
            for _ in range(self.config.concurrency):
                await queue.put(None)

        async def worker() -> None:
            while True:
                job = await queue.get()
                if job is None:
                    return
                seed, sample_index = job
                if await self._budget_exhausted():
                    self._stop_scheduling.set()
                    self.stats.skipped += 1
                    continue
                await self._generate_one(seed, sample_index)

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(produce())
            for _ in range(self.config.concurrency):
                tasks.create_task(worker())
        return self.stats

    async def _generate_one(self, seed: SeedTask, sample_index: int) -> None:
        key = job_key(seed, sample_index, self.config.model)
        messages = build_messages(seed, sample_index)
        request_seed = deterministic_seed(key)
        started = time.monotonic()
        try:
            result = await self.client.complete(
                model=self.config.model,
                fallbacks=self.config.fallbacks,
                messages=messages,
                response_format=KERNEL_RESPONSE_SCHEMA,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=request_seed,
                reasoning_effort=self.config.reasoning_effort,
                data_collection=self.config.data_collection,
                zdr=self.config.zdr,
            )
        except OpenRouterError as exc:
            async with self._write_lock:
                append_jsonl(
                    self.config.rejected_output,
                    {
                        "job_key": key,
                        "seed_id": seed.id,
                        "sample_index": sample_index,
                        "status": "request_failed",
                        "error": str(exc),
                    },
                )
                self.stats.failed += 1
            return

        raw_row = {
            "job_key": key,
            "seed_id": seed.id,
            "sample_index": sample_index,
            "requested_model": result.requested_model,
            "actual_model": result.actual_model,
            "provider": result.provider,
            "response_id": result.response_id,
            "usage": result.usage,
            "content": result.content,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

        try:
            candidate = parse_candidate(result.content)
            validation = validate_candidate(candidate, seed)
        except ValueError as exc:
            candidate = None
            validation = None
            parse_error = str(exc)
        else:
            parse_error = None

        async with self._write_lock:
            append_jsonl(self.config.raw_output, raw_row)
            self.stats.cost_usd += result.cost

            if parse_error:
                append_jsonl(
                    self.config.rejected_output,
                    {
                        **raw_row,
                        "status": "invalid_json",
                        "issues": [parse_error],
                    },
                )
                self.stats.rejected += 1
                return

            assert candidate is not None and validation is not None
            issues = list(validation.issues)
            if validation.solution_hash in self._seen_hashes:
                issues.append("duplicate solution in current run")
            if issues:
                append_jsonl(
                    self.config.rejected_output,
                    {
                        **raw_row,
                        "status": "static_validation_failed",
                        "issues": issues,
                        "candidate": candidate,
                    },
                )
                self.stats.rejected += 1
                return

            self._seen_hashes.add(validation.solution_hash)
            record_id = stable_record_id(seed.id, sample_index, validation.solution_hash)
            append_jsonl(
                self.config.output,
                {
                    "id": record_id,
                    "job_key": key,
                    "source": "synthetic-openrouter",
                    "seed_id": seed.id,
                    "sample_index": sample_index,
                    "operation": seed.operation,
                    "backend": seed.backend,
                    "instruction": candidate["instruction"],
                    "reference_code": seed.reference_code,
                    "solution_code": candidate["solution_code"],
                    "messages": [
                        {"role": "system", "content": messages[0]["content"]},
                        {"role": "user", "content": messages[1]["content"]},
                        {"role": "assistant", "content": render_assistant_text(candidate)},
                    ],
                    "teacher": {
                        "requested_model": result.requested_model,
                        "actual_model": result.actual_model,
                        "provider": result.provider,
                        "response_id": result.response_id,
                    },
                    "generation": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "max_tokens": self.config.max_tokens,
                        "reasoning_effort": self.config.reasoning_effort,
                        "request_seed": request_seed,
                        "usage": result.usage,
                    },
                    "quality": {
                        "static_validation": "passed",
                        "gpu_validation": "not_run",
                        "solution_sha256": validation.solution_hash,
                    },
                    "metadata": seed.metadata,
                },
            )
            self.stats.accepted += 1

    async def _budget_exhausted(self) -> bool:
        if self.config.budget_usd is None:
            return False
        async with self._budget_lock:
            return self.stats.cost_usd >= self.config.budget_usd


def deterministic_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF


def generation_jobs(
    seeds: list[SeedTask],
    *,
    samples_per_seed: int,
    target_rows: int | None = None,
):
    """Yield jobs evenly across seeds without building a large in-memory list."""
    if target_rows is None:
        target_rows = len(seeds) * samples_per_seed

    produced = 0
    sample_index = 0
    while produced < target_rows:
        for seed in seeds:
            if produced >= target_rows:
                return
            yield seed, sample_index
            produced += 1
        sample_index += 1


def job_key(seed: SeedTask, sample_index: int, model: str) -> str:
    payload = json.dumps(
        {
            "seed_id": seed.id,
            "sample_index": sample_index,
            "model": model,
            "backend": seed.backend,
            "reference_sha256": hashlib.sha256(seed.reference_code.encode()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
