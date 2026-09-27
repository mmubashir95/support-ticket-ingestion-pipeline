"""Tests for the frozen classification split configuration."""

import pytest
from pydantic import ValidationError

from ticket_classification.splitting import SplitConfig


DATASET_VERSION = "ds_6fa2f19cf683"


def test_valid_split_configuration_is_accepted() -> None:
    config = SplitConfig(
        dataset_version=DATASET_VERSION,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify=True,
    )

    assert config.dataset_version == DATASET_VERSION
    assert config.train_ratio == 0.70
    assert config.validation_ratio == 0.15
    assert config.test_ratio == 0.15
    assert config.random_seed == 42
    assert config.stratify is True


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio", "test_ratio"),
    [
        (0.70, 0.20, 0.20),
        (0.50, 0.20, 0.10),
    ],
    ids=["total-greater-than-one", "total-less-than-one"],
)
def test_split_ratios_must_total_one(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    with pytest.raises(ValidationError, match="must total 1.0"):
        SplitConfig(
            dataset_version=DATASET_VERSION,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("train_ratio", -0.10),
        ("validation_ratio", 0.0),
        ("test_ratio", 1.0),
        ("test_ratio", 1.5),
    ],
    ids=["negative", "zero", "equal-to-one", "greater-than-one"],
)
def test_each_split_ratio_must_be_between_zero_and_one(
    field_name: str,
    invalid_value: float,
) -> None:
    values = {field_name: invalid_value}

    with pytest.raises(ValidationError):
        SplitConfig(dataset_version=DATASET_VERSION, **values)


@pytest.mark.parametrize("dataset_version", ["", "   "])
def test_dataset_version_must_not_be_blank(dataset_version: str) -> None:
    with pytest.raises(ValidationError, match="non-blank string"):
        SplitConfig(dataset_version=dataset_version)


def test_split_configuration_defaults_are_frozen() -> None:
    config = SplitConfig(dataset_version=DATASET_VERSION)

    assert config.train_ratio == 0.70
    assert config.validation_ratio == 0.15
    assert config.test_ratio == 0.15
    assert config.random_seed == 42
    assert config.stratify is True


def test_split_configuration_json_round_trip_preserves_values() -> None:
    config = SplitConfig(dataset_version=DATASET_VERSION)

    reloaded = SplitConfig.model_validate_json(config.model_dump_json())

    assert reloaded == config
