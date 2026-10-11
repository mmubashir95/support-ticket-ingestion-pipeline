"""CLI for the Phase 2.12 final model comparison."""

import argparse
from pathlib import Path

from ticket_classification.benchmarking import DEFAULT_RUN_ROOT
from ticket_classification.final_comparison import (
    DEFAULT_OUTPUT_DIR, build_final_comparison, write_final_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final Phase 2.12 three-model comparison")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    try:
        data = build_final_comparison(args.run_root, bootstrap_repetitions=args.bootstrap_repetitions)
        _, report = write_final_comparison(data, args.output_dir)
        print(report)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(2, f"Phase 2.12 error: {error}\n")


if __name__ == "__main__":
    main()
