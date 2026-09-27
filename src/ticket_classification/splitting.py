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


class SplitIntegrityError(ValueError):
    """A produced split violates the frozen split contract."""


@dataclass(frozen=True, slots=True)
class SplitIntegrityReport:
    """Compact evidence that an in-memory split passed integrity validation."""

    dataset_version: str
    expected_total: int
    actual_total: int
    train_count: int
    validation_count: int
    test_count: int
    overlap_count: int
    missing_count: int
    extra_count: int
    duplicate_assignment_count: int
    class_counts: dict[str, dict[str, int]]


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

    train_count, validation_count, _ = _calculate_split_counts(
        len(shuffled), config
    )
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
    train_count, validation_count, test_count = _calculate_split_counts(total, config)
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


def validate_split_integrity(
    records: Sequence[ClassificationRecord],
    config: SplitConfig,
    result: SplitResult,
    *,
    expected_dataset_version: str,
) -> SplitIntegrityReport:
    """Validate that a split is complete, disjoint, and policy compliant."""

    if not records:
        raise SplitIntegrityError("input records must not be empty")
    if config.dataset_version != expected_dataset_version:
        raise SplitIntegrityError(
            "dataset version mismatch: "
            f"config={config.dataset_version!r}, expected={expected_dataset_version!r}"
        )

    original_ids = [record.record_id for record in records]
    if len(original_ids) != len(set(original_ids)):
        raise SplitIntegrityError("input records must have unique record_id values")

    groups = {
        "train": result.train,
        "validation": result.validation,
        "test": result.test,
    }
    group_ids = {
        name: [record.record_id for record in group]
        for name, group in groups.items()
    }
    group_id_sets = {name: set(ids) for name, ids in group_ids.items()}

    overlap_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    overlaps = {
        f"{left}/{right}": group_id_sets[left] & group_id_sets[right]
        for left, right in overlap_pairs
        if group_id_sets[left] & group_id_sets[right]
    }
    if overlaps:
        details = "; ".join(
            f"{pair} overlap={len(ids)}, sample={sorted(ids)[:5]}"
            for pair, ids in overlaps.items()
        )
        raise SplitIntegrityError(f"split groups overlap: {details}")

    assigned_ids = [record_id for ids in group_ids.values() for record_id in ids]
    unique_assigned_ids = set(assigned_ids)
    duplicate_count = len(assigned_ids) - len(unique_assigned_ids)
    if duplicate_count:
        duplicate_ids = sorted(
            record_id
            for record_id, count in Counter(assigned_ids).items()
            if count > 1
        )
        raise SplitIntegrityError(
            "duplicate split assignment: "
            f"duplicate_count={duplicate_count}, sample={duplicate_ids[:5]}"
        )

    original_id_set = set(original_ids)
    missing_ids = original_id_set - unique_assigned_ids
    if missing_ids:
        raise SplitIntegrityError(
            "split is missing input records: "
            f"expected={len(original_ids)}, actual_unique={len(unique_assigned_ids)}, "
            f"missing={len(missing_ids)}, sample={sorted(missing_ids)[:5]}"
        )

    extra_ids = unique_assigned_ids - original_id_set
    if extra_ids:
        raise SplitIntegrityError(
            "split contains foreign records: "
            f"extra={len(extra_ids)}, sample={sorted(extra_ids)[:5]}"
        )

    expected_counts = _calculate_split_counts(len(records), config)
    actual_counts = tuple(len(groups[name]) for name in groups)
    if actual_counts != expected_counts:
        raise SplitIntegrityError(
            "split counts do not match configured allocation: "
            f"expected train/validation/test={expected_counts}, actual={actual_counts}"
        )
    if len(assigned_ids) != len(records):
        raise SplitIntegrityError(
            "total split count does not match input count: "
            f"expected={len(records)}, actual={len(assigned_ids)}"
        )

    if any(not isinstance(record.label, str) for record in records):
        raise SplitIntegrityError(
            "split integrity validation requires one string label per record"
        )
    class_counts = _class_counts(records, groups)
    if config.stratify:
        _validate_stratification(records, groups, class_counts)

    return SplitIntegrityReport(
        dataset_version=config.dataset_version,
        expected_total=len(records),
        actual_total=len(assigned_ids),
        train_count=len(result.train),
        validation_count=len(result.validation),
        test_count=len(result.test),
        overlap_count=0,
        missing_count=0,
        extra_count=0,
        duplicate_assignment_count=0,
        class_counts=class_counts,
    )


def _calculate_split_counts(total: int, config: SplitConfig) -> tuple[int, int, int]:
    train_count = math.floor(total * config.train_ratio)
    validation_count = math.floor(total * config.validation_ratio)
    test_count = total - train_count - validation_count
    return train_count, validation_count, test_count


def _class_counts(
    records: Sequence[ClassificationRecord],
    groups: dict[str, list[ClassificationRecord]],
) -> dict[str, dict[str, int]]:
    all_groups: dict[str, Sequence[ClassificationRecord]] = {"full": records, **groups}
    return {
        name: dict(sorted(Counter(record.label for record in group).items()))
        for name, group in all_groups.items()
    }


def _validate_stratification(
    records: Sequence[ClassificationRecord],
    groups: dict[str, list[ClassificationRecord]],
    class_counts: dict[str, dict[str, int]],
) -> None:
    if any(not isinstance(record.label, str) for record in records):
        raise SplitIntegrityError(
            "stratification integrity requires one string label per record"
        )

    full_labels = set(class_counts["full"])
    for name in groups:
        group_labels = set(class_counts[name])
        missing_labels = full_labels - group_labels
        if missing_labels:
            raise SplitIntegrityError(
                f"stratified {name} split is missing classes: "
                f"{sorted(missing_labels)}"
            )

    for name, group in groups.items():
        # Two-stage stratification can accumulate at most approximately two
        # records of integer-allocation error for a class. Expressing that as
        # 2 / split size makes the percentage tolerance scale with the sample.
        proportion_tolerance = 2 / len(group)
        for label in sorted(full_labels):
            full_proportion = class_counts["full"][label] / len(records)
            split_proportion = class_counts[name][label] / len(group)
            if abs(split_proportion - full_proportion) > proportion_tolerance:
                raise SplitIntegrityError(
                    f"stratified {name} class distribution is invalid for "
                    f"{label!r}: full={full_proportion:.6f}, "
                    f"split={split_proportion:.6f}, "
                    f"tolerance={proportion_tolerance:.6f}"
                )
