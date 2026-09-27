"""Validated configuration for future classification dataset splitting."""

import math
import random
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
    """Assign records using a deterministic, non-stratified random shuffle.

    Train and validation counts are floored from their configured ratios. The
    test group receives every remaining record, ensuring exact coverage even
    when the ratios do not produce whole-number counts.
    """

    if config.stratify:
        raise ValueError(
            "stratify=True is not supported by the basic splitter; "
            "stratification belongs to Phase 2.2.5"
        )
    if not records:
        raise ValueError("records must contain at least one classification record")

    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records must have unique record_id values")

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
