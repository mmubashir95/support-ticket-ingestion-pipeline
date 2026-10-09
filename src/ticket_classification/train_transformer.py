"""CLI for Phase 2.8 validation, smoke testing, and user-run training."""

import argparse
from pathlib import Path

from ticket_classification.transformer_training import run_training, validate_only


DEFAULT_DATASET = Path("artifacts/classification/run/transformer")
DEFAULT_OUTPUT = Path("artifacts/classification/run/transformer_training")
DEFAULT_SMOKE_OUTPUT = Path("artifacts/classification/run/transformer_training_smoke")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the frozen DistilBERT baseline")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="load and forward-check without training")
    modes.add_argument("--smoke-test", action="store_true", help="run 32/16-example integration training")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        validate_only(args.dataset_dir, args.output_dir or DEFAULT_OUTPUT)
    elif args.smoke_test:
        run_training(args.dataset_dir, args.output_dir or DEFAULT_SMOKE_OUTPUT, smoke_test=True)
    else:
        run_training(args.dataset_dir, args.output_dir or DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
