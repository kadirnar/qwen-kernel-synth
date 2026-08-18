from kernel_synth.prompts import build_messages, mutation_profile
from kernel_synth.types import SeedTask


def test_mutation_profile_is_deterministic() -> None:
    assert mutation_profile("seed", 3) == mutation_profile("seed", 3)


def test_prompt_contains_reference_and_anti_contamination_rule() -> None:
    seed = SeedTask(
        id="x",
        operation="add",
        objective="fuse addition",
        reference_code="class Model: pass",
        backend="triton",
    )
    messages = build_messages(seed, 0)

    assert messages[0]["role"] == "system"
    assert "Do not reproduce" in messages[0]["content"]
    assert "class Model: pass" in messages[1]["content"]
    assert "Required backend: triton" in messages[1]["content"]
