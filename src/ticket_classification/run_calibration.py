"""Manual Phase 2.10 CLI. Commands are validation-only and never retrain a base model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ticket_classification.calibration_workflow import (
    MODEL_NAMES,
    analyze_uncalibrated_models,
    audit_prediction_artifacts,
    fit_model_calibration,
    generate_selected_validation_predictions,
    generate_final_policies,
    render_final_report,
    run_threshold_analysis,
)
from ticket_classification.thresholds import DEFAULT_THRESHOLDS


DEFAULT_ROOT = Path("artifacts/classification/run")
DEFAULT_OUTPUT = DEFAULT_ROOT / "threshold_calibration"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)


def _models(value: str) -> tuple[str, ...]:
    return MODEL_NAMES if value == "all" else (value,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explicit Phase 2.10 calibration and threshold steps"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="audit selected validation predictions")
    _common(audit)

    analyze = commands.add_parser(
        "analyze", help="create subsets and evaluate uncalibrated confidence"
    )
    _common(analyze)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--bins", type=int, default=10)

    generate = commands.add_parser(
        "generate-predictions",
        help="regenerate missing validation predictions without retraining",
    )
    _common(generate)
    generate.add_argument(
        "--model", choices=(*MODEL_NAMES, "all"), default="all"
    )

    fit = commands.add_parser("fit", help="fit post-hoc calibration parameters")
    _common(fit)
    fit.add_argument(
        "--model", choices=(*MODEL_NAMES, "all"), default="all"
    )
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--bins", type=int, default=10)

    thresholds = commands.add_parser(
        "thresholds", help="evaluate confidence/coverage trade-offs"
    )
    _common(thresholds)
    thresholds.add_argument(
        "--model", choices=(*MODEL_NAMES, "all"), default="all"
    )
    thresholds.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )

    compare = commands.add_parser(
        "compare", help="select calibration and persist threshold policy"
    )
    _common(compare)
    compare.add_argument(
        "--min-selective-accuracy",
        type=float,
        default=None,
        help="business criterion; omit to leave operational thresholds pending",
    )

    report = commands.add_parser("report", help="render the final Phase 2.10 report")
    _common(report)

    args = parser.parse_args()
    if args.command == "audit":
        audit_prediction_artifacts(args.run_root, args.output_dir)
    elif args.command == "generate-predictions":
        for model in _models(args.model):
            generate_selected_validation_predictions(
                args.run_root, args.output_dir, model
            )
    elif args.command == "analyze":
        analyze_uncalibrated_models(
            args.run_root,
            args.output_dir,
            bins=args.bins,
            seed=args.seed,
        )
    elif args.command == "fit":
        for model in _models(args.model):
            fit_model_calibration(
                args.run_root,
                args.output_dir,
                model,
                bins=args.bins,
                seed=args.seed,
            )
    elif args.command == "thresholds":
        run_threshold_analysis(
            args.run_root,
            args.output_dir,
            _models(args.model),
            thresholds=args.thresholds,
        )
    elif args.command == "compare":
        generate_final_policies(
            args.run_root,
            args.output_dir,
            minimum_selective_accuracy=args.min_selective_accuracy,
        )
    else:
        render_final_report(args.output_dir)


if __name__ == "__main__":
    main()
