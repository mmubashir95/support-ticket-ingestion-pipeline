"""Tests for persisted and frozen classification split artifacts."""

from collections import Counter
from pathlib import Path

import pytest

from ticket_classification.models import ClassificationRecord
from ticket_classification.split_outputs import (
    SPLIT_MANIFEST_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VALIDATION_FILENAME,
    SplitArtifacts,
    load_classification_records,
    load_split_manifest,
    write_split_artifacts,
)
from ticket_classification.splitting import (
    SplitConfig,
    SplitIntegrityError,
    SplitResult,
    split_classification_records,
)


DATASET_VERSION = "ds_test_fixture"


def records(count: int = 80) -> list[ClassificationRecord]:
    return [
        ClassificationRecord(
            record_id=f"record:{index:03d}",
            text=f"Ticket text {index}",
            label=f"class-{index % 4}",
            source_fields=["subject", "message"],
        )
        for index in range(count)
    ]


def frozen_split() -> tuple[list[ClassificationRecord], SplitConfig, SplitResult]:
    source = records()
    config = SplitConfig(dataset_version=DATASET_VERSION)
    return source, config, split_classification_records(source, config)


def write_fixture(output_dir: Path) -> tuple[SplitArtifacts, list[ClassificationRecord], SplitConfig, SplitResult]:
    source, config, result = frozen_split()
    artifacts = write_split_artifacts(
        source,
        result,
        config,
        output_dir,
        expected_dataset_version=DATASET_VERSION,
    )
    return artifacts, source, config, result


def ids(group: list[ClassificationRecord]) -> list[str]:
    return [record.record_id for record in group]


def test_valid_split_writes_all_four_parseable_artifacts(tmp_path: Path) -> None:
    artifacts, _, _, _ = write_fixture(tmp_path)

    assert isinstance(artifacts, SplitArtifacts)
    assert artifacts.train_path == tmp_path / TRAIN_FILENAME
    assert artifacts.validation_path == tmp_path / VALIDATION_FILENAME
    assert artifacts.test_path == tmp_path / TEST_FILENAME
    assert artifacts.manifest_path == tmp_path / SPLIT_MANIFEST_FILENAME
    assert all(
        path.is_file()
        for path in (
            artifacts.train_path,
            artifacts.validation_path,
            artifacts.test_path,
            artifacts.manifest_path,
        )
    )
    assert load_classification_records(artifacts.train_path)
    assert load_classification_records(artifacts.validation_path)
    assert load_classification_records(artifacts.test_path)
    assert load_split_manifest(artifacts.manifest_path)


def test_persisted_counts_and_ids_match_split_result(tmp_path: Path) -> None:
    artifacts, _, _, result = write_fixture(tmp_path)
    loaded_groups = (
        load_classification_records(artifacts.train_path),
        load_classification_records(artifacts.validation_path),
        load_classification_records(artifacts.test_path),
    )

    assert tuple(len(group) for group in loaded_groups) == (
        len(result.train),
        len(result.validation),
        len(result.test),
    )
    assert tuple(ids(group) for group in loaded_groups) == (
        ids(result.train),
        ids(result.validation),
        ids(result.test),
    )


def test_manifest_freezes_counts_class_counts_and_membership(tmp_path: Path) -> None:
    artifacts, source, _, result = write_fixture(tmp_path)
    manifest = load_split_manifest(artifacts.manifest_path)

    assert manifest.total_count == len(source)
    assert manifest.train_count == len(result.train)
    assert manifest.validation_count == len(result.validation)
    assert manifest.test_count == len(result.test)
    assert manifest.class_counts_full == dict(
        Counter(record.label for record in source)
    )
    assert manifest.class_counts_train == dict(
        Counter(record.label for record in result.train)
    )
    assert manifest.class_counts_validation == dict(
        Counter(record.label for record in result.validation)
    )
    assert manifest.class_counts_test == dict(
        Counter(record.label for record in result.test)
    )
    assert manifest.train_record_ids == ids(result.train)
    assert manifest.validation_record_ids == ids(result.validation)
    assert manifest.test_record_ids == ids(result.test)


def test_manifest_freezes_dataset_version_seed_ratios_and_policy(tmp_path: Path) -> None:
    artifacts, _, config, _ = write_fixture(tmp_path)
    manifest = load_split_manifest(artifacts.manifest_path)

    assert manifest.dataset_version == config.dataset_version
    assert manifest.random_seed == config.random_seed
    assert manifest.train_ratio == config.train_ratio
    assert manifest.validation_ratio == config.validation_ratio
    assert manifest.test_ratio == config.test_ratio
    assert manifest.stratified is config.stratify
    assert manifest.split_policy.allocation == "floor/floor/remainder"
    assert manifest.split_policy.stratification_key == "label"


def test_invalid_split_is_not_persisted(tmp_path: Path) -> None:
    source, config, result = frozen_split()
    invalid = SplitResult(
        train=result.train,
        validation=[*result.validation, result.train[0]],
        test=result.test,
    )

    with pytest.raises(SplitIntegrityError, match="overlap"):
        write_split_artifacts(
            source,
            invalid,
            config,
            tmp_path / "invalid",
            expected_dataset_version=DATASET_VERSION,
        )

    assert not (tmp_path / "invalid").exists()


def test_same_split_produces_byte_identical_artifacts(tmp_path: Path) -> None:
    first, _, _, _ = write_fixture(tmp_path / "first")
    second, _, _, _ = write_fixture(tmp_path / "second")

    first_paths = (
        first.train_path,
        first.validation_path,
        first.test_path,
        first.manifest_path,
    )
    second_paths = (
        second.train_path,
        second.validation_path,
        second.test_path,
        second.manifest_path,
    )
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]


def test_persisted_ids_are_disjoint_and_cover_all_inputs(tmp_path: Path) -> None:
    artifacts, source, _, _ = write_fixture(tmp_path)
    train_ids = set(ids(load_classification_records(artifacts.train_path)))
    validation_ids = set(ids(load_classification_records(artifacts.validation_path)))
    test_ids = set(ids(load_classification_records(artifacts.test_path)))

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(ids(source))
