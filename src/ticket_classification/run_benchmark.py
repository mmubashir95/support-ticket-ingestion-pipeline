"""Command-line entry point for Phase 2.11 (benchmark execution is opt-in)."""

import argparse
import json
from pathlib import Path

from ticket_classification.benchmarking import (
    DEFAULT_RUN_ROOT, BenchmarkConfig, regenerate_report, run_benchmarks,
    validate_artifacts, write_results,
)


DEFAULT_ROOT = DEFAULT_RUN_ROOT


def _configuration(args) -> BenchmarkConfig:
    costs = {}
    for item in args.hourly_cost:
        device, value = item.split("=", 1)
        if device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"unknown cost device: {device}")
        costs[device] = float(value)
    return BenchmarkConfig(
        run_root=args.run_root, output_dir=args.output_dir, models=tuple(args.models),
        transformer_devices=tuple(args.transformer_devices),
        include_available_accelerator=args.include_available_accelerator,
        batch_sizes=tuple(args.batch_sizes), warmup_iterations=args.warmup_iterations,
        repetitions=args.repetitions, sample_count=args.sample_count,
        random_seed=args.seed, split=args.split, hourly_costs=costs,
    )


def _config_arguments(parser):
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "inference_benchmark")
    parser.add_argument("--models", nargs="+", choices=("logistic_regression", "linear_svm", "transformer"), default=("logistic_regression", "linear_svm", "transformer"))
    parser.add_argument("--transformer-devices", nargs="+", choices=("cpu", "cuda", "mps"), default=("cpu",))
    parser.add_argument("--include-available-accelerator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 8, 16, 32, 64))
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--hourly-cost", action="append", default=[], metavar="DEVICE=RATE")


def main():
    parser = argparse.ArgumentParser(description="Phase 2.11 batch inference benchmarking")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        _config_arguments(commands.add_parser(name))
    report = commands.add_parser("report")
    report.add_argument("--results", type=Path, default=DEFAULT_ROOT / "inference_benchmark" / "benchmark_results.json")
    report.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "report":
            print(regenerate_report(args.results, args.output))
            return
        config = _configuration(args)
        if args.command == "check":
            print(json.dumps(validate_artifacts(config), indent=2, sort_keys=True))
            return
        results = run_benchmarks(config)
        write_results(config, results)
        print(config.output_dir / "inference_benchmark.md")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Phase 2.11 error: {error}\n")


if __name__ == "__main__":
    main()
