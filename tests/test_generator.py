import asyncio
import json
from pathlib import Path

from kernel_synth.generator import (
    DatasetGenerator,
    GenerationConfig,
    generation_jobs,
    job_key,
)
from kernel_synth.types import CompletionResult, SeedTask

SOLUTION = """import torch
import triton
import triton.language as tl

@triton.jit
def add_one_kernel(x, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + 1.0)

class ModelNew(torch.nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        add_one_kernel[(1,)](x, out, n=x.numel())
        return out

def get_inputs():
    return [torch.randn(16, device="cuda")]

def get_init_inputs():
    return []
"""


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, **_: object) -> CompletionResult:
        self.calls += 1
        content = json.dumps(
            {
                "instruction": "Implement a custom kernel that adds one to every tensor value.",
                "backend": "triton",
                "reasoning_summary": "Use one coalesced launch for this bandwidth-bound operation.",
                "estimated_bottleneck": "memory bandwidth",
                "optimization_notes": ["coalesced reads", "single device launch"],
                "assumptions": ["input is contiguous"],
                "test_strategy": "Compare against PyTorch on random values and boundary sizes.",
                "solution_code": SOLUTION,
            }
        )
        return CompletionResult(
            content=content,
            response_id="generation-1",
            requested_model="teacher/model",
            actual_model="teacher/model",
            provider="test",
            usage={"cost": 0.01, "prompt_tokens": 10, "completion_tokens": 20},
            raw_response={},
        )


def test_generator_writes_auditable_record_and_resumes(tmp_path: Path) -> None:
    seed = SeedTask(
        id="add-one",
        operation="add one",
        objective="fuse an elementwise add",
        reference_code="class Model: pass",
        backend="triton",
    )
    config = GenerationConfig(
        model="teacher/model",
        fallbacks=(),
        output=tmp_path / "accepted.jsonl",
        rejected_output=tmp_path / "rejected.jsonl",
        raw_output=tmp_path / "raw.jsonl",
    )
    client = FakeClient()

    first = asyncio.run(DatasetGenerator(client=client, config=config).run([seed]))
    second = asyncio.run(DatasetGenerator(client=client, config=config).run([seed]))

    row = json.loads(config.output.read_text(encoding="utf-8"))
    assert first.accepted == 1
    assert second.skipped == 1
    assert client.calls == 1
    assert row["job_key"] == job_key(seed, 0, "teacher/model")
    assert row["teacher"]["actual_model"] == "teacher/model"
    assert row["quality"]["gpu_validation"] == "not_run"


def test_raw_row_alone_does_not_mark_job_complete(tmp_path: Path) -> None:
    seed = SeedTask(
        id="resume-after-crash",
        operation="add one",
        objective="fuse an elementwise add",
        reference_code="class Model: pass",
        backend="triton",
    )
    config = GenerationConfig(
        model="teacher/model",
        fallbacks=(),
        output=tmp_path / "accepted.jsonl",
        rejected_output=tmp_path / "rejected.jsonl",
        raw_output=tmp_path / "raw.jsonl",
    )
    config.raw_output.write_text(
        json.dumps({"job_key": job_key(seed, 0, config.model)}) + "\n",
        encoding="utf-8",
    )
    client = FakeClient()

    stats = asyncio.run(DatasetGenerator(client=client, config=config).run([seed]))

    assert stats.accepted == 1
    assert client.calls == 1


def test_target_rows_are_distributed_across_seeds() -> None:
    seeds = [
        SeedTask(
            id=seed_id,
            operation="add one",
            objective="fuse an elementwise add",
            reference_code="class Model: pass",
        )
        for seed_id in ("a", "b")
    ]

    jobs = list(generation_jobs(seeds, samples_per_seed=1, target_rows=5))

    assert [(seed.id, index) for seed, index in jobs] == [
        ("a", 0),
        ("b", 0),
        ("a", 1),
        ("b", 1),
        ("a", 2),
    ]
