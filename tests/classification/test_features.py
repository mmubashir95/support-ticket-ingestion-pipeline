"""Tests for the train-only sparse TF-IDF feature pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.sparse import csr_matrix, issparse

from ticket_classification.feature_outputs import (
    build_feature_config,
    build_feature_metadata,
    load_feature_config,
    load_feature_metadata,
    load_sparse_features,
    load_vectorizer,
    write_feature_artifacts,
)
from ticket_classification.features import (
    CLASSIFICATION_INPUT_FIELDS,
    CLASSIFICATION_INPUT_POLICY,
    CLASSIFICATION_TEXT_SEPARATOR,
    FrozenSplits,
    TfidfConfig,
    build_classification_text,
    build_tfidf_features,
    fit_tfidf,
    load_frozen_splits,
    transform_tfidf,
)
from ticket_classification.models import ClassificationRecord
from ticket_classification.split_outputs import (
    SplitManifest,
    SplitPolicy,
    serialize_classification_records,
    split_manifest_to_json,
)


def record(index: int, text: str, label: str = "Incident") -> ClassificationRecord:
    return ClassificationRecord(
        record_id=f"record:{index}",
        text=text,
        label=label,
        source_fields=["subject", "message"],
    )


def frozen_splits() -> FrozenSplits:
    train = [
        record(1, "apple account login"),
        record(2, "banana account password", "Request"),
        record(3, "apple banana access"),
        record(4, "password login access", "Request"),
    ]
    validation = [record(5, "validationonlyword account")]
    test = [record(6, "supersecretword login", "Request")]
    manifest = SplitManifest(
        dataset_version="ds_fixture",
        target="ticket_type",
        random_seed=42,
        split_policy=SplitPolicy(stratification_key="label"),
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        stratified=True,
        total_count=6,
        train_count=4,
        validation_count=1,
        test_count=1,
        class_counts_full={"Incident": 3, "Request": 3},
        class_counts_train={"Incident": 2, "Request": 2},
        class_counts_validation={"Incident": 1},
        class_counts_test={"Request": 1},
        train_record_ids=[item.record_id for item in train],
        validation_record_ids=[item.record_id for item in validation],
        test_record_ids=[item.record_id for item in test],
    )
    return FrozenSplits(
        train=train,
        validation=validation,
        test=test,
        manifest=manifest,
    )


def test_frozen_input_text_policy_is_explicit_and_deterministic() -> None:
    item = record(1, "Cannot access account\n\nPassword reset did not work")

    assert CLASSIFICATION_INPUT_POLICY == "subject + message"
    assert CLASSIFICATION_INPUT_FIELDS == ("subject", "message")
    assert CLASSIFICATION_TEXT_SEPARATOR == "\n\n"
    assert build_classification_text(item) == item.text
    assert build_classification_text(item) == build_classification_text(item)


@pytest.mark.parametrize("text", ["Only message", "Only subject", ""])
def test_frozen_text_preserves_phase_2_1_empty_field_handling(text: str) -> None:
    assert build_classification_text(record(1, text)) == text


def test_text_policy_rejects_unexpected_source_fields() -> None:
    item = record(1, "text").model_copy(update={"source_fields": ["message"]})

    with pytest.raises(ValueError, match="source_fields"):
        build_classification_text(item)


def test_default_tfidf_configuration_is_frozen() -> None:
    config = TfidfConfig()

    assert config.analyzer == "word"
    assert config.ngram_range == (1, 2)
    assert config.lowercase is True
    assert config.min_df == 2
    assert config.max_df is None
    assert config.sublinear_tf is False
    assert config.max_features is None


@pytest.mark.parametrize(
    "values",
    [
        {"ngram_range": (0, 2)},
        {"ngram_range": (2, 1)},
        {"min_df": 0},
        {"min_df": 1.1},
        {"max_df": 0},
        {"max_df": 1.1},
        {"min_df": 3, "max_df": 2},
        {"min_df": 0.8, "max_df": 0.5},
        {"max_features": 0},
        {"max_features": -1},
    ],
)
def test_invalid_tfidf_configuration_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TfidfConfig(**values)


def test_train_only_fit_blocks_validation_and_test_vocabulary_leakage() -> None:
    splits = frozen_splits()
    result = build_tfidf_features(splits, TfidfConfig(min_df=1))
    changed_holdouts = FrozenSplits(
        train=splits.train,
        validation=[record(5, "entirely changed validation vocabulary")],
        test=[record(6, "entirely changed test vocabulary", "Request")],
        manifest=splits.manifest,
    )
    changed_result = build_tfidf_features(changed_holdouts, TfidfConfig(min_df=1))

    assert "validationonlyword" not in result.vectorizer.vocabulary_
    assert "supersecretword" not in result.vectorizer.vocabulary_
    assert result.vectorizer.vocabulary_ == changed_result.vectorizer.vocabulary_
    np.testing.assert_allclose(result.vectorizer.idf_, changed_result.vectorizer.idf_)
    assert result.validation.shape[0] == 1
    assert result.test.shape[0] == 1


def test_outputs_are_sparse_csr_and_share_one_feature_space() -> None:
    splits = frozen_splits()
    result = build_tfidf_features(splits, TfidfConfig(min_df=1))
    matrices = (result.train, result.validation, result.test)

    assert all(issparse(matrix) for matrix in matrices)
    assert all(isinstance(matrix, csr_matrix) for matrix in matrices)
    assert {matrix.shape[1] for matrix in matrices} == {
        len(result.vectorizer.vocabulary_)
    }
    assert [matrix.shape[0] for matrix in matrices] == [4, 1, 1]


def test_transform_does_not_expand_fitted_vocabulary() -> None:
    splits = frozen_splits()
    vectorizer, _ = fit_tfidf(splits.train, TfidfConfig(min_df=1))
    before = dict(vectorizer.vocabulary_)

    transform_tfidf(vectorizer, splits.validation)
    transform_tfidf(vectorizer, splits.test)

    assert vectorizer.vocabulary_ == before


def test_same_training_records_and_config_produce_identical_features() -> None:
    splits = frozen_splits()
    config = TfidfConfig(min_df=1)
    first = build_tfidf_features(splits, config)
    second = build_tfidf_features(splits, config)

    assert first.vectorizer.vocabulary_ == second.vectorizer.vocabulary_
    np.testing.assert_allclose(first.vectorizer.idf_, second.vectorizer.idf_)
    assert (first.train != second.train).nnz == 0
    assert (first.validation != second.validation).nnz == 0
    assert (first.test != second.test).nnz == 0


def test_feature_metadata_records_source_shapes_and_nonzero_counts() -> None:
    splits = frozen_splits()
    config = TfidfConfig(min_df=1)
    result = build_tfidf_features(splits, config)
    metadata = build_feature_metadata(splits, result, build_feature_config(config))

    assert metadata.dataset_version == splits.manifest.dataset_version
    assert metadata.split_version.startswith("split_")
    assert metadata.feature_version.startswith("tfidf_")
    assert metadata.fit_split == "train"
    assert metadata.vocabulary_size == len(result.vectorizer.vocabulary_)
    assert metadata.feature_count == result.train.shape[1]
    assert metadata.matrix_shapes == {
        "train": result.train.shape,
        "validation": result.validation.shape,
        "test": result.test.shape,
    }
    assert metadata.non_zero_counts == {
        "train": result.train.nnz,
        "validation": result.validation.nnz,
        "test": result.test.nnz,
    }


def test_vectorizer_sparse_matrices_and_json_round_trip(tmp_path: Path) -> None:
    splits = frozen_splits()
    config = TfidfConfig(min_df=1)
    result = build_tfidf_features(splits, config)
    artifacts = write_feature_artifacts(splits, result, config, tmp_path)

    reloaded_vectorizer = load_vectorizer(artifacts.vectorizer_path)
    original_transform = transform_tfidf(result.vectorizer, splits.test)
    reloaded_transform = transform_tfidf(reloaded_vectorizer, splits.test)
    assert (original_transform != reloaded_transform).nnz == 0

    paths_and_expected = (
        (artifacts.train_features_path, result.train),
        (artifacts.validation_features_path, result.validation),
        (artifacts.test_features_path, result.test),
    )
    for path, expected in paths_and_expected:
        loaded = load_sparse_features(path)
        assert loaded.shape == expected.shape
        assert loaded.nnz == expected.nnz
        assert (loaded != expected).nnz == 0

    assert load_feature_config(artifacts.config_path) == build_feature_config(config)
    assert load_feature_metadata(artifacts.metadata_path).dataset_version == "ds_fixture"


def test_frozen_split_loader_preserves_manifest_row_order(tmp_path: Path) -> None:
    splits = frozen_splits()
    (tmp_path / "train.jsonl").write_text(
        serialize_classification_records(splits.train), encoding="utf-8"
    )
    (tmp_path / "validation.jsonl").write_text(
        serialize_classification_records(splits.validation), encoding="utf-8"
    )
    (tmp_path / "test.jsonl").write_text(
        serialize_classification_records(splits.test), encoding="utf-8"
    )
    (tmp_path / "split_manifest.json").write_text(
        split_manifest_to_json(splits.manifest) + "\n", encoding="utf-8"
    )

    loaded = load_frozen_splits(tmp_path)

    assert [item.record_id for item in loaded.train] == splits.manifest.train_record_ids
    assert [item.record_id for item in loaded.validation] == splits.manifest.validation_record_ids
    assert [item.record_id for item in loaded.test] == splits.manifest.test_record_ids


def test_frozen_split_loader_rejects_manifest_order_mismatch(tmp_path: Path) -> None:
    splits = frozen_splits()
    (tmp_path / "train.jsonl").write_text(
        serialize_classification_records(list(reversed(splits.train))), encoding="utf-8"
    )
    (tmp_path / "validation.jsonl").write_text(
        serialize_classification_records(splits.validation), encoding="utf-8"
    )
    (tmp_path / "test.jsonl").write_text(
        serialize_classification_records(splits.test), encoding="utf-8"
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps(splits.manifest.model_dump(mode="json")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="record order"):
        load_frozen_splits(tmp_path)
