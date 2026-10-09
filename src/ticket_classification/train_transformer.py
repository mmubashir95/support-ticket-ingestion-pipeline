"""CLI for unweighted or Phase 2.9 weighted Transformer training."""

import argparse
from pathlib import Path

from ticket_classification.transformer_training import (
    TransformerTrainingConfig,
    run_training,
    validate_only,
)


DEFAULT_DATASET = Path("artifacts/classification/run/transformer")
DEFAULT_OUTPUT = Path("artifacts/classification/run/transformer_training")
DEFAULT_SMOKE_OUTPUT = Path("artifacts/classification/run/transformer_training_smoke")
DEFAULT_WEIGHTED_OUTPUT = Path("artifacts/classification/run/class_weighting/transformer")
DEFAULT_WEIGHTED_SMOKE_OUTPUT = Path("artifacts/classification/run/class_weighting/transformer_smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the frozen DistilBERT baseline")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="load and forward-check without training")
    modes.add_argument("--smoke-test", action="store_true", help="run 32/16-example integration training")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--class-weighting", choices=("none", "balanced"), default="none",
        help="use train-only balanced weighted cross-entropy",
    )
    args = parser.parse_args()
    config = TransformerTrainingConfig(class_weighting=args.class_weighting)
    default_output = DEFAULT_WEIGHTED_OUTPUT if args.class_weighting == "balanced" else DEFAULT_OUTPUT
    default_smoke = (
        DEFAULT_WEIGHTED_SMOKE_OUTPUT if args.class_weighting == "balanced" else DEFAULT_SMOKE_OUTPUT
    )
    if args.validate_only:
        validate_only(args.dataset_dir, args.output_dir or default_output, config)
    elif args.smoke_test:
        run_training(args.dataset_dir, args.output_dir or default_smoke, config, smoke_test=True)
    else:
        run_training(args.dataset_dir, args.output_dir or default_output, config)


if __name__ == "__main__":
    main()
