"""Validated configuration for future classification dataset splitting."""

import math

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
