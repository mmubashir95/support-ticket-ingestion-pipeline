"""Tests for the frozen classification split configuration."""

from collections import Counter

import pytest
from pydantic import ValidationError

from ticket_classification.models import ClassificationRecord
from ticket_classification.splitting import SplitConfig, split_classification_records


DATASET_VERSION = "ds_6fa2f19cf683"


def records(count: int = 20) -> list[ClassificationRecord]:
    return [
        ClassificationRecord(
            record_id=f"record:{index:03d}",
            text=f"Ticket text {index}",
            label=f"class-{index % 4}",
            source_fields=["subject", "message"],
        )
        for index in range(count)
    ]


def basic_config(seed: int = 42) -> SplitConfig:
    return SplitConfig(
        dataset_version=DATASET_VERSION,
        random_seed=seed,
        stratify=False,
    )


def stratified_config(seed: int = 42) -> SplitConfig:
    return SplitConfig(dataset_version=DATASET_VERSION, random_seed=seed)


def split_ids(group: list[ClassificationRecord]) -> list[str]:
    return [record.record_id for record in group]


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


def test_same_seed_produces_same_split_assignments() -> None:
    source = records(40)

    first = split_classification_records(source, basic_config(seed=42))
    second = split_classification_records(source, basic_config(seed=42))

    assert split_ids(first.train) == split_ids(second.train)
    assert split_ids(first.validation) == split_ids(second.validation)
    assert split_ids(first.test) == split_ids(second.test)


def test_split_does_not_mutate_original_record_order() -> None:
    source = records()
    original_ids = split_ids(source)

    split_classification_records(source, basic_config())

    assert split_ids(source) == original_ids


def test_split_groups_do_not_overlap_and_cover_every_record() -> None:
    source = records(23)

    result = split_classification_records(source, basic_config())

    train_ids = set(split_ids(result.train))
    validation_ids = set(split_ids(result.validation))
    test_ids = set(split_ids(result.test))
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(split_ids(source))
    assert len(result.train) + len(result.validation) + len(result.test) == len(source)


def test_split_uses_floor_counts_and_assigns_remainder_to_test() -> None:
    result = split_classification_records(records(23), basic_config())

    assert len(result.train) == 16
    assert len(result.validation) == 3
    assert len(result.test) == 4


def test_twenty_records_have_expected_split_counts() -> None:
    result = split_classification_records(records(20), basic_config())

    assert len(result.train) == 14
    assert len(result.validation) == 3
    assert len(result.test) == 3


def test_different_seeds_normally_change_assignments() -> None:
    source = records(40)

    seed_42 = split_classification_records(source, basic_config(seed=42))
    seed_7 = split_classification_records(source, basic_config(seed=7))

    assert split_ids(seed_42.train) != split_ids(seed_7.train)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        split_classification_records([], basic_config())


def test_duplicate_record_ids_are_rejected() -> None:
    source = records(2)
    source[1] = source[1].model_copy(update={"record_id": source[0].record_id})

    with pytest.raises(ValueError, match="unique record_id"):
        split_classification_records(source, basic_config())


def test_stratified_split_is_supported() -> None:
    result = split_classification_records(records(80), stratified_config())

    assert len(result.train) == 56
    assert len(result.validation) == 12
    assert len(result.test) == 12


def test_same_seed_produces_same_stratified_assignments() -> None:
    source = records(80)

    first = split_classification_records(source, stratified_config(seed=42))
    second = split_classification_records(source, stratified_config(seed=42))

    assert split_ids(first.train) == split_ids(second.train)
    assert split_ids(first.validation) == split_ids(second.validation)
    assert split_ids(first.test) == split_ids(second.test)


def test_every_class_appears_in_every_stratified_split() -> None:
    source = records(80)
    result = split_classification_records(source, stratified_config())
    expected_labels = {record.label for record in source}

    assert {record.label for record in result.train} == expected_labels
    assert {record.label for record in result.validation} == expected_labels
    assert {record.label for record in result.test} == expected_labels


def test_stratified_class_proportions_stay_close_to_full_dataset() -> None:
    source = records(200)
    result = split_classification_records(source, stratified_config())
    full_counts = Counter(record.label for record in source)

    for group in (result.train, result.validation, result.test):
        group_counts = Counter(record.label for record in group)
        for label, full_count in full_counts.items():
            full_proportion = full_count / len(source)
            group_proportion = group_counts[label] / len(group)
            assert group_proportion == pytest.approx(full_proportion, abs=0.02)


def test_stratified_split_has_no_overlap_and_complete_coverage() -> None:
    source = records(83)
    result = split_classification_records(source, stratified_config())

    train_ids = set(split_ids(result.train))
    validation_ids = set(split_ids(result.validation))
    test_ids = set(split_ids(result.test))
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(split_ids(source))
    assert len(result.train) + len(result.validation) + len(result.test) == len(source)


def test_stratified_split_does_not_mutate_input_order() -> None:
    source = records(80)
    original_ids = split_ids(source)

    split_classification_records(source, stratified_config())

    assert split_ids(source) == original_ids


def test_stratified_split_with_different_seeds_changes_assignments() -> None:
    source = records(200)

    seed_42 = split_classification_records(source, stratified_config(seed=42))
    seed_7 = split_classification_records(source, stratified_config(seed=7))

    assert split_ids(seed_42.train) != split_ids(seed_7.train)
    assert Counter(record.label for record in seed_42.train) == Counter(
        record.label for record in seed_7.train
    )


def test_too_small_class_fails_stratification_clearly() -> None:
    source = records(40)
    source[-1] = source[-1].model_copy(update={"label": "too-small"})

    with pytest.raises(ValueError, match="'too-small' has 1"):
        split_classification_records(source, stratified_config())
