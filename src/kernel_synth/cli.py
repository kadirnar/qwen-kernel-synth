from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .catalog import fetch_recommended_models
from .client import AsyncOpenRouterClient
from .generator import DatasetGenerator, GenerationConfig
from .io import load_seeds, read_jsonl
from .types import SeedTask
from .validation import validate_candidate

DEFAULT_MODEL = "z-ai/glm-5.2"
DEFAULT_FALLBACKS = ("moonshotai/kimi-k2.7-code", "deepseek/deepseek-v4-flash-0731")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernel-synth",
        description="Generate auditable CUDA/Triton SFT data with OpenRouter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate synthetic JSONL records")
    generate.add_argument("--seeds", type=Path, required=True, help="input seed-task JSONL")
    generate.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    generate.add_argument("--model", default=DEFAULT_MODEL)
    generate.add_argument(
        "--fallback",
        action="append",
        default=None,
        help="fallback model slug; repeat the flag for multiple fallbacks",
    )
    generate.add_argument("--samples-per-seed", type=_positive_int, default=1)
    generate.add_argument(
        "--target-rows",
        type=_positive_int,
        help="exact number of candidate requests; overrides --samples-per-seed",
    )
    generate.add_argument("--concurrency", type=_positive_int, default=4)
    generate.add_argument("--max-tokens", type=_positive_int, default=8192)
    generate.add_argument("--temperature", type=float, default=0.45)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "max", "xhigh"),
        default="high",
    )
    generate.add_argument("--budget-usd", type=_positive_float)
    generate.add_argument("--timeout", type=_positive_float, default=180.0)
    generate.add_argument("--max-retries", type=_nonnegative_int, default=5)
    generate.add_argument(
        "--allow-provider-data-collection",
        action="store_true",
        help="allow providers that may retain data; default is deny",
    )
    generate.add_argument(
        "--zdr",
        action="store_true",
        help="require a Zero Data Retention provider endpoint",
    )

    validate = subparsers.add_parser("validate", help="re-run static checks on accepted JSONL")
    validate.add_argument("--input", type=Path, required=True)

    subparsers.add_parser("models", help="show live pricing for recommended OpenRouter models")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            asyncio.run(_generate(args))
        elif args.command == "validate":
            _validate(args.input)
        elif args.command == "models":
            _models()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


async def _generate(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("set OPENROUTER_API_KEY before generating data")
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be greater than 0 and at most 1")

    seeds = load_seeds(args.seeds)
    output_dir: Path = args.output_dir
    config = GenerationConfig(
        model=args.model,
        fallbacks=tuple(args.fallback if args.fallback is not None else DEFAULT_FALLBACKS),
        output=output_dir / "accepted.jsonl",
        rejected_output=output_dir / "rejected.jsonl",
        raw_output=output_dir / "raw.jsonl",
        samples_per_seed=args.samples_per_seed,
        target_rows=args.target_rows,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
        budget_usd=args.budget_usd,
        data_collection="allow" if args.allow_provider_data_collection else "deny",
        zdr=args.zdr,
    )
    async with AsyncOpenRouterClient(
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        site_url=os.environ.get("OPENROUTER_SITE_URL"),
        app_name=os.environ.get("OPENROUTER_APP_NAME", "qwen-kernel-synth"),
    ) as client:
        stats = await DatasetGenerator(client=client, config=config).run(seeds)

    print(
        json.dumps(
            {
                "requested": stats.requested,
                "skipped": stats.skipped,
                "accepted": stats.accepted,
                "rejected": stats.rejected,
                "failed": stats.failed,
                "reported_cost_usd": round(stats.cost_usd, 6),
                "output": str(config.output),
            },
            indent=2,
        )
    )


def _validate(path: Path) -> None:
    failures = 0
    rows = read_jsonl(path)
    for index, row in enumerate(rows, start=1):
        seed = SeedTask.from_mapping(
            {
                "id": row.get("seed_id", f"row-{index}"),
                "operation": row.get("operation", "unknown"),
                "objective": row.get("instruction", "validate existing generated record"),
                "reference_code": row.get("reference_code", "# missing"),
                "backend": row.get("backend", "triton"),
            }
        )
        candidate = {
            "instruction": row.get("instruction", ""),
            "backend": row.get("backend", ""),
            "reasoning_summary": "existing accepted record re-validation",
            "estimated_bottleneck": "unknown",
            "optimization_notes": ["existing record", "static validation"],
            "assumptions": [],
            "test_strategy": "execute against the reference over randomized inputs",
            "solution_code": row.get("solution_code", ""),
        }
        result = validate_candidate(candidate, seed)
        if not result.accepted:
            failures += 1
            print(f"row {index} ({row.get('id', '?')}): {'; '.join(result.issues)}")
    print(f"validated={len(rows)} failed={failures}")
    if failures:
        raise SystemExit(1)


def _models() -> None:
    rows = fetch_recommended_models()
    headings = ("MODEL", "ROLE", "CONTEXT", "INPUT $/M", "OUTPUT $/M", "JSON SCHEMA")
    print("\t".join(headings))
    for row in rows:
        print(
            "\t".join(
                str(value)
                for value in (
                    row["id"],
                    row["role"],
                    row["context_length"],
                    row["input_per_million"],
                    row["output_per_million"],
                    row["structured_outputs"],
                )
            )
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    main()
