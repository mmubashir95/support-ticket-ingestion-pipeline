"""Manual Phase 2.9 workflow; each command is explicit and validation-only."""

import argparse
from pathlib import Path

from ticket_classification.class_weighting import (
    run_class_imbalance_analysis,
    run_weighted_classical_experiment,
    run_weighting_comparison,
)


DEFAULT_ROOT = Path("artifacts/classification/run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2.9 class-weighting steps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "classical"):
        child = subparsers.add_parser(name)
        child.add_argument("--split-dir", type=Path, default=DEFAULT_ROOT / "splits")
        child.add_argument("--feature-dir", type=Path, default=DEFAULT_ROOT / "tfidf")
        child.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "class_weighting")
        if name == "classical":
            child.add_argument(
                "--model", choices=("logistic_regression", "linear_svm", "all"), default="all"
            )
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline-root", type=Path, default=DEFAULT_ROOT)
    compare.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "class_weighting")
    args = parser.parse_args()
    if args.command == "analyze":
        run_class_imbalance_analysis(
            args.split_dir, args.feature_dir, args.baseline_root, args.output_dir
        )
    elif args.command == "classical":
        models = ("logistic_regression", "linear_svm") if args.model == "all" else (args.model,)
        for model in models:
            run_weighted_classical_experiment(
                model,
                args.split_dir,
                args.feature_dir,
                args.baseline_root / model,
                args.output_dir / model,
            )
    else:
        run_weighting_comparison(args.baseline_root, args.output_dir)


if __name__ == "__main__":
    main()
