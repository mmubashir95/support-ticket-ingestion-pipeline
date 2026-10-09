"""Separate final-test inference CLI for the selected DistilBERT checkpoint."""

import argparse
from pathlib import Path

from ticket_classification.transformer_training import run_final_test_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected Transformer on frozen test")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/classification/run/transformer"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/classification/run/transformer_training"))
    args = parser.parse_args()
    run_final_test_evaluation(args.dataset_dir, args.training_dir)


if __name__ == "__main__":
    main()
