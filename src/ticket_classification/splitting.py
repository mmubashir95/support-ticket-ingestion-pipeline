"""Validated configuration for future classification dataset splitting."""

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from sklearn.model_selection import train_test_split

from ticket_classification.models import ClassificationRecord


@dataclass(frozen=True, slots=True)
class SplitResult:
    """One in-memory assignment of classification records to split groups."""

    train: list[ClassificationRecord]
    validation: list[ClassificationRecord]
    test: list[ClassificationRecord]


class SplitConfig(BaseModel):
    """Frozen policy for a future train/validation/test split.

    This model only describes and validates the split policy. Record assignment
    belongs to the deterministic splitting step that follows this one.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr
    train_ratio: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_seed: StrictInt = 42
    stratify: StrictBool = True

    @field_validator("dataset_version")
    @classmethod
    def validate_dataset_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dataset_version must be a non-blank string")
        return value

    @model_validator(mode="after")
    def validate_ratio_total(self) -> "SplitConfig":
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "train_ratio, validation_ratio, and test_ratio must total 1.0 "
                f"(received {total:.12g})"
            )
        return self


def split_classification_records(
    records: Sequence[ClassificationRecord],
    config: SplitConfig,
) -> SplitResult:
    """Assign records using a deterministic random or stratified split.

    Train and validation counts are floored from their configured ratios. The
    test group receives every remaining record, ensuring exact coverage even
    when the ratios do not produce whole-number counts.
    """

    if not records:
        raise ValueError("records must contain at least one classification record")

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records must have unique record_id values")

    if config.stratify:
        return _stratified_split(records, config)

    shuffled = list(records)
    random.Random(config.random_seed).shuffle(shuffled)

    train_count = math.floor(len(shuffled) * config.train_ratio)
    validation_count = math.floor(len(shuffled) * config.validation_ratio)
    validation_end = train_count + validation_count

    return SplitResult(
        train=shuffled[:train_count],
        validation=shuffled[train_count:validation_end],
        test=shuffled[validation_end:],
    )


def _stratified_split(
    records: Sequence[ClassificationRecord],
    config: SplitConfig,
) -> SplitResult:
    labels: list[str] = []
    for record in records:
        if not isinstance(record.label, str):
            raise ValueError(
                "stratified splitting requires one string label per record"
            )
        labels.append(record.label)

    class_counts = Counter(labels)
    too_small = {
        label: count for label, count in class_counts.items() if count < 3
    }
    if too_small:
        details = ", ".join(
            f"{label!r} has {count}" for label, count in sorted(too_small.items())
        )
        raise ValueError(
            "stratified splitting requires at least 3 records per class so each "
            f"class can appear in train, validation, and test; {details}"
        )

    total = len(records)
    train_count = math.floor(total * config.train_ratio)
    validation_count = math.floor(total * config.validation_ratio)
    test_count = total - train_count - validation_count
    class_count = len(class_counts)
    split_counts = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    undersized_splits = [
        name for name, count in split_counts.items() if count < class_count
    ]
    if undersized_splits:
        names = ", ".join(undersized_splits)
        raise ValueError(
            "stratified splitting cannot place every class in each split: "
            f"{names} contain fewer records than the {class_count} classes"
        )

    seed_generator = random.Random(config.random_seed)
    first_seed = seed_generator.randrange(2**32)
    second_seed = seed_generator.randrange(2**32)
    try:
        train, temporary = train_test_split(
            list(records),
            train_size=train_count,
            test_size=validation_count + test_count,
            random_state=first_seed,
            shuffle=True,
            stratify=labels,
        )
        temporary_labels = [record.label for record in temporary]
        validation, test = train_test_split(
            temporary,
            train_size=validation_count,
            test_size=test_count,
            random_state=second_seed,
            shuffle=True,
            stratify=temporary_labels,
        )
    except ValueError as error:
        counts = ", ".join(
            f"{label!r}={count}" for label, count in sorted(class_counts.items())
        )
        raise ValueError(
            "stratified splitting is not possible for the configured ratios and "
            f"class counts ({counts}): {error}"
        ) from error

    return SplitResult(train=train, validation=validation, test=test)
