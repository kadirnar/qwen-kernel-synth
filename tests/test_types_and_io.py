import json
from pathlib import Path

import pytest

from kernel_synth.io import completed_job_keys, load_seeds
from kernel_synth.types import SeedTask


def test_seed_mapping_validation() -> None:
    seed = SeedTask.from_mapping(
        {
            "id": "a",
            "operation": "sum",
            "objective": "optimize a reduction",
            "reference_code": "class Model: pass",
            "backend": "triton",
            "constraints": ["float32"],
        }
    )
    assert seed.constraints == ("float32",)


def test_seed_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        SeedTask.from_mapping(
            {
                "id": "a",
                "operation": "sum",
                "objective": "optimize",
                "reference_code": "pass",
                "backend": "metal",
            }
        )


def test_load_seeds_and_resume_keys(tmp_path: Path) -> None:
    seeds_path = tmp_path / "seeds.jsonl"
    seeds_path.write_text(
        json.dumps(
            {
                "id": "a",
                "operation": "sum",
                "objective": "optimize",
                "reference_code": "class Model: pass",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"
    output.write_text('{"job_key":"done"}\n', encoding="utf-8")

    assert load_seeds(seeds_path)[0].id == "a"
    assert completed_job_keys(output) == {"done"}
