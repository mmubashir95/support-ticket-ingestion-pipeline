"""Tests for the frozen classification split configuration."""

from collections import Counter

import pytest
from pydantic import ValidationError

from ticket_classification.models import ClassificationRecord
from ticket_classification.splitting import (
    SplitIntegrityError,
    SplitResult,
    SplitConfig,
    split_classification_records,
    validate_split_integrity,
)


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


def validate(
    source: list[ClassificationRecord],
    config: SplitConfig,
    result: SplitResult,
    expected_dataset_version: str = DATASET_VERSION,
):
    return validate_split_integrity(
        source,
        config,
        result,
        expected_dataset_version=expected_dataset_version,
    )


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
    assert config.target_field == "ticket_type"


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


@pytest.mark.parametrize("target_field", ["", "   "])
def test_target_field_must_not_be_blank(target_field: str) -> None:
    with pytest.raises(ValidationError, match="non-blank string"):
        SplitConfig(dataset_version=DATASET_VERSION, target_field=target_field)


def test_split_configuration_defaults_are_frozen() -> None:
    config = SplitConfig(dataset_version=DATASET_VERSION)

    assert config.train_ratio == 0.70
    assert config.validation_ratio == 0.15
    assert config.test_ratio == 0.15
    assert config.random_seed == 42
    assert config.stratify is True
    assert config.target_field == "ticket_type"


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


def test_valid_stratified_split_passes_integrity_validation() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)

    report = validate(source, config, result)

    assert report.dataset_version == DATASET_VERSION
    assert report.expected_total == 80
    assert report.actual_total == 80
    assert (report.train_count, report.validation_count, report.test_count) == (
        56,
        12,
        12,
    )
    assert report.overlap_count == 0
    assert report.missing_count == 0
    assert report.extra_count == 0
    assert report.duplicate_assignment_count == 0


@pytest.mark.parametrize(
    ("source_group", "destination_group", "expected_pair"),
    [
        ("train", "validation", "train/validation"),
        ("train", "test", "train/test"),
        ("validation", "test", "validation/test"),
    ],
)
def test_split_overlap_fails_integrity_validation(
    source_group: str,
    destination_group: str,
    expected_pair: str,
) -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    groups = {
        "train": list(result.train),
        "validation": list(result.validation),
        "test": list(result.test),
    }
    groups[destination_group].append(groups[source_group][0])
    tampered = SplitResult(**groups)

    with pytest.raises(SplitIntegrityError, match=expected_pair):
        validate(source, config, tampered)


def test_missing_record_fails_integrity_validation() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    tampered = SplitResult(
        train=result.train,
        validation=result.validation,
        test=result.test[:-1],
    )

    with pytest.raises(SplitIntegrityError, match=r"missing=1"):
        validate(source, config, tampered)


def test_duplicate_assignment_within_one_group_fails() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    tampered = SplitResult(
        train=[*result.train, result.train[0]],
        validation=result.validation,
        test=result.test,
    )

    with pytest.raises(SplitIntegrityError, match=r"duplicate_count=1"):
        validate(source, config, tampered)


def test_foreign_record_fails_integrity_validation() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    foreign = ClassificationRecord(
        record_id="record:foreign",
        text="Foreign ticket",
        label="class-0",
        source_fields=["subject", "message"],
    )
    tampered = SplitResult(
        train=[*result.train, foreign],
        validation=result.validation,
        test=result.test,
    )

    with pytest.raises(SplitIntegrityError, match=r"foreign records: extra=1"):
        validate(source, config, tampered)


def test_wrong_individual_split_counts_fail() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    tampered = SplitResult(
        train=[*result.train, result.test[0]],
        validation=result.validation,
        test=result.test[1:],
    )

    with pytest.raises(SplitIntegrityError, match="counts do not match"):
        validate(source, config, tampered)


def test_dataset_version_mismatch_fails_integrity_validation() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)

    with pytest.raises(SplitIntegrityError, match="dataset version mismatch"):
        validate(source, config, result, expected_dataset_version="ds_other")


def test_missing_class_in_stratified_split_fails() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    target_label = result.validation[0].label
    validation_targets = [
        record for record in result.validation if record.label == target_label
    ]
    train_replacements = [
        record for record in result.train if record.label != target_label
    ][: len(validation_targets)]
    target_ids = set(split_ids(validation_targets))
    replacement_ids = set(split_ids(train_replacements))
    tampered = SplitResult(
        train=[
            *(record for record in result.train if record.record_id not in replacement_ids),
            *validation_targets,
        ],
        validation=[
            *(record for record in result.validation if record.record_id not in target_ids),
            *train_replacements,
        ],
        test=result.test,
    )

    with pytest.raises(SplitIntegrityError, match="is missing classes"):
        validate(source, config, tampered)


def test_broken_stratified_proportions_fail() -> None:
    source = [
        ClassificationRecord(
            record_id=f"binary:{index:03d}",
            text=f"Ticket {index}",
            label="A" if index < 100 else "B",
            source_fields=["subject", "message"],
        )
        for index in range(200)
    ]
    config = stratified_config()
    result = split_classification_records(source, config)
    validation_b = [record for record in result.validation if record.label == "B"]
    train_a = [record for record in result.train if record.label == "A"]
    validation_to_move = validation_b[:-1]
    train_to_move = train_a[: len(validation_to_move)]
    validation_move_ids = set(split_ids(validation_to_move))
    train_move_ids = set(split_ids(train_to_move))
    tampered = SplitResult(
        train=[
            *(record for record in result.train if record.record_id not in train_move_ids),
            *validation_to_move,
        ],
        validation=[
            *(
                record
                for record in result.validation
                if record.record_id not in validation_move_ids
            ),
            *train_to_move,
        ],
        test=result.test,
    )

    with pytest.raises(SplitIntegrityError, match="class distribution is invalid"):
        validate(source, config, tampered)


def test_non_stratified_integrity_does_not_require_class_preservation() -> None:
    source = [
        ClassificationRecord(
            record_id=f"non-stratified:{index:02d}",
            text=f"Ticket {index}",
            label="A" if index < 10 else "B",
            source_fields=["subject", "message"],
        )
        for index in range(20)
    ]
    config = basic_config()
    deliberately_skewed = SplitResult(
        train=source[:14],
        validation=source[14:17],
        test=source[17:],
    )

    report = validate(source, config, deliberately_skewed)

    assert set(report.class_counts["test"]) == {"B"}


def test_integrity_validation_preserves_input_and_split_order() -> None:
    source = records(80)
    config = stratified_config()
    result = split_classification_records(source, config)
    original_source_ids = split_ids(source)
    original_group_ids = (
        split_ids(result.train),
        split_ids(result.validation),
        split_ids(result.test),
    )

    validate(source, config, result)

    assert split_ids(source) == original_source_ids
    assert (
        split_ids(result.train),
        split_ids(result.validation),
        split_ids(result.test),
    ) == original_group_ids
