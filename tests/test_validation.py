from kernel_synth.types import SeedTask
from kernel_synth.validation import stable_record_id, validate_candidate

GOOD_SOLUTION = """import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x, y, out, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(out + offsets, tl.load(x + offsets) + tl.load(y + offsets))

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        add_kernel[(1,)](x, y, out, n=x.numel())
        return out

def get_inputs():
    return [torch.randn(16, device="cuda"), torch.randn(16, device="cuda")]

def get_init_inputs():
    return []
"""


def candidate(solution: str = GOOD_SOLUTION) -> dict[str, object]:
    return {
        "instruction": "Write a custom kernel for elementwise tensor addition.",
        "backend": "triton",
        "reasoning_summary": "Use one coalesced program because the operation is bandwidth bound.",
        "estimated_bottleneck": "memory bandwidth",
        "optimization_notes": ["coalesced loads", "one fused launch"],
        "assumptions": ["contiguous inputs"],
        "test_strategy": "Compare with PyTorch over random inputs and boundary sizes.",
        "solution_code": solution,
    }


def seed() -> SeedTask:
    return SeedTask(
        id="add",
        operation="add",
        objective="fuse add",
        reference_code="class Model: pass",
        backend="triton",
    )


def test_accepts_structurally_valid_triton_program() -> None:
    result = validate_candidate(candidate(), seed())
    assert result.accepted
    assert len(result.solution_hash) == 64


def test_rejects_network_and_torch_compile() -> None:
    bad = GOOD_SOLUTION + "\nimport requests\ntorch.compile(ModelNew())\n"
    result = validate_candidate(candidate(bad), seed())
    assert not result.accepted
    assert any("network access" in issue for issue in result.issues)
    assert any("torch.compile" in issue for issue in result.issues)


def test_record_id_is_stable() -> None:
    assert stable_record_id("a", 1, "hash") == stable_record_id("a", 1, "hash")
