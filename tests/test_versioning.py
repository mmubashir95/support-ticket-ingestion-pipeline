from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

import pytest
from pydantic import ValidationError

from ticket_pipeline.language import LanguageDetectionConfig
from ticket_pipeline.leakage import LeakageConfig
from ticket_pipeline.models import DatasetSourceManifest, SourceFileManifest
from ticket_pipeline.versioning import (
    canonical_json,
    canonical_json_bytes,
    capture_processing_config,
    compare_dataset_manifests,
    create_dataset_manifest,
    fingerprint_records,
    fingerprint_source_file,
    fingerprint_source_files,
    fingerprint_value,
    get_pipeline_version,
    load_dataset_manifest,
    sha256_hex,
    write_dataset_manifest,
)


class ExampleStatus(Enum):
    READY = "ready"


@dataclass(frozen=True)
class ExampleConfig:
    fields: frozenset[str]
    enabled: bool


def source_manifest(name: str = "tickets.csv", digest: str = "a" * 64):
    return DatasetSourceManifest(
        files=[
            SourceFileManifest(
                name=name,
                sha256=digest,
                size_bytes=10,
            )
        ]
    )


def processing_config(*, threshold: float = 0.85):
    return capture_processing_config(
        semantic_deduplication_enabled=True,
        semantic_threshold=threshold,
        language_config=LanguageDetectionConfig(),
        leakage_config=LeakageConfig(
            forbidden_fields={"resolution_text", "answer"},
            target_fields={"priority"},
            group_fields={"conversation_id"},
        ),
    )


def make_manifest(
    *,
    source: DatasetSourceManifest | None = None,
    config_threshold: float = 0.85,
    records: list[object] | None = None,
    pipeline_version: str = "0.1.0",
    created_at: datetime | None = None,
    input_record_count: int = 2,
):
    return create_dataset_manifest(
        source=source or source_manifest(),
        processing_config=processing_config(threshold=config_threshold),
        input_record_count=input_record_count,
        processed_records=(
            records
            if records is not None
            else [{"message": "first"}, {"message": "second"}]
        ),
        pipeline_version=pipeline_version,
        created_at=created_at or datetime(2026, 9, 2, tzinfo=timezone.utc),
    )


def test_sha256_helper_uses_stable_cryptographic_hash() -> None:
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_canonical_json_sorts_mapping_keys_and_uses_compact_separators() -> None:
    first = canonical_json({"b": 2, "a": 1})
    second = canonical_json({"a": 1, "b": 2})

    assert first == '{"a":1,"b":2}'
    assert first == second
    assert canonical_json_bytes({"b": 2, "a": 1}) == first.encode("utf-8")


def test_canonical_json_handles_enums_dataclasses_and_sets() -> None:
    value = {
        "status": ExampleStatus.READY,
        "config": ExampleConfig(
            fields=frozenset({"closed_at", "answer"}),
            enabled=True,
        ),
    }

    serialized = canonical_json(value)

    assert serialized == (
        '{"config":{"enabled":true,"fields":["answer","closed_at"]},'
        '"status":"ready"}'
    )


def test_set_order_does_not_change_fingerprint() -> None:
    assert fingerprint_value({"fields": {"a", "b", "c"}}) == (
        fingerprint_value({"fields": {"c", "a", "b"}})
    )


def test_list_order_does_change_fingerprint() -> None:
    assert fingerprint_value(["a", "b"]) != fingerprint_value(["b", "a"])


def test_unicode_is_normalized_for_structured_fingerprints() -> None:
    assert fingerprint_value({"name": "José"}) == fingerprint_value(
        {"name": "Jose\u0301"}
    )


def test_mapping_keys_that_collide_after_unicode_normalization_fail() -> None:
    with pytest.raises(ValueError, match="collide"):
        canonical_json({"é": 1, "e\u0301": 2})


def test_equal_instants_have_equal_datetime_serialization() -> None:
    utc_value = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)
    offset_value = datetime(
        2026,
        9,
        2,
        15,
        0,
        tzinfo=timezone(timedelta(hours=5)),
    )

    assert fingerprint_value(utc_value) == fingerprint_value(offset_value)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        canonical_json(datetime(2026, 9, 2, 10, 0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floats_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json(value)


def test_arbitrary_objects_are_not_serialized_with_repr() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        canonical_json(object())


def test_record_fingerprint_preserves_ingestion_order() -> None:
    records = [{"message": "first"}, {"message": "second"}]

    assert fingerprint_records(records) != fingerprint_records(list(reversed(records)))
    assert fingerprint_records(records) == fingerprint_records(records)


def test_source_file_hash_uses_exact_bytes_and_stable_name(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    source.write_bytes(b"subject,body\nLogin,Cannot login\n")

    result = fingerprint_source_file(source)

    assert result.name == "tickets.csv"
    assert result.sha256 == sha256_hex(source.read_bytes())
    assert result.size_bytes == len(source.read_bytes())
    assert str(tmp_path) not in result.model_dump_json()


def test_source_byte_or_order_change_changes_file_hash(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    source.write_bytes(b"first\nsecond\n")
    first = fingerprint_source_file(source)
    source.write_bytes(b"second\nfirst\n")
    second = fingerprint_source_file(source)

    assert first.sha256 != second.sha256


def test_multiple_source_files_are_sorted_independent_of_argument_order(
    tmp_path,
) -> None:
    first_path = tmp_path / "a.csv"
    second_path = tmp_path / "b.json"
    first_path.write_text("a", encoding="utf-8")
    second_path.write_text("b", encoding="utf-8")

    first = fingerprint_source_files([second_path, first_path])
    second = fingerprint_source_files([first_path, second_path])

    assert [item.name for item in first.files] == ["a.csv", "b.json"]
    assert first == second
    assert fingerprint_value(first.files) == fingerprint_value(second.files)


def test_duplicate_source_names_are_rejected() -> None:
    duplicate = SourceFileManifest(
        name="tickets.csv",
        sha256="a" * 64,
        size_bytes=1,
    )

    with pytest.raises(ValidationError, match="unique"):
        DatasetSourceManifest(files=[duplicate, duplicate])


def test_machine_specific_source_path_is_rejected_from_manifest() -> None:
    with pytest.raises(ValidationError, match="file name"):
        SourceFileManifest(
            name="/tmp/tickets.csv",
            sha256="a" * 64,
            size_bytes=1,
        )


def test_source_name_is_unicode_normalized() -> None:
    source = SourceFileManifest(
        name="cafe\u0301.csv",
        sha256="a" * 64,
        size_bytes=1,
    )

    assert source.name == "café.csv"


def test_explicit_blank_source_name_is_rejected(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    source.write_text("data", encoding="utf-8")

    with pytest.raises(ValidationError):
        fingerprint_source_file(source, source_name="")


def test_processing_config_captures_existing_material_settings() -> None:
    config = processing_config(threshold=0.91)

    assert config.semantic_deduplication is not None
    assert config.semantic_deduplication.threshold == 0.91
    assert config.language_detection.confidence_threshold == 0.8
    assert config.language_detection.min_alphabetic_characters == 4
    assert "ENGLISH" in config.language_detection.supported_languages
    assert config.leakage.forbidden_fields == ["answer", "resolution_text"]
    assert config.leakage.target_fields == ["priority"]


def test_disabled_semantic_settings_do_not_affect_config_snapshot() -> None:
    first = capture_processing_config(
        semantic_deduplication_enabled=False,
        semantic_threshold=0.80,
    )
    second = capture_processing_config(
        semantic_deduplication_enabled=False,
        semantic_threshold=0.99,
    )

    assert first.semantic_deduplication is None
    assert first == second


def test_same_content_and_config_produce_same_dataset_identity() -> None:
    first = make_manifest(
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc)
    )
    second = make_manifest(
        created_at=datetime(2026, 9, 3, tzinfo=timezone.utc)
    )

    assert first.dataset_version == second.dataset_version
    assert first.fingerprints == second.fingerprints
    assert first.created_at != second.created_at
    assert first.dataset_version.startswith("ds_")
    assert len(first.dataset_version) == 15


def test_changed_source_changes_input_and_dataset_identity() -> None:
    first = make_manifest(source=source_manifest(digest="a" * 64))
    second = make_manifest(source=source_manifest(digest="b" * 64))

    assert first.fingerprints.input != second.fingerprints.input
    assert first.dataset_version != second.dataset_version


def test_changed_config_changes_config_and_dataset_identity() -> None:
    first = make_manifest(config_threshold=0.85)
    second = make_manifest(config_threshold=0.90)

    assert first.fingerprints.config != second.fingerprints.config
    assert first.dataset_version != second.dataset_version


def test_changed_output_changes_output_and_dataset_identity() -> None:
    first = make_manifest(records=[{"message": "first"}])
    second = make_manifest(records=[{"message": "changed"}])

    assert first.fingerprints.output != second.fingerprints.output
    assert first.dataset_version != second.dataset_version


def test_changed_pipeline_version_changes_dataset_identity() -> None:
    first = make_manifest(pipeline_version="0.1.0")
    second = make_manifest(pipeline_version="0.2.0")

    assert first.fingerprints == second.fingerprints
    assert first.dataset_version != second.dataset_version


def test_explicit_blank_pipeline_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_manifest(pipeline_version="")


def test_manifest_counts_only_currently_available_values() -> None:
    manifest = make_manifest(
        records=[{"message": "one"}],
        input_record_count=2,
    )

    assert manifest.counts.input == 2
    assert manifest.counts.processed == 1
    assert "accepted" not in manifest.counts.model_dump()
    assert "rejected" not in manifest.counts.model_dump()


def test_processed_count_cannot_exceed_input_count() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        make_manifest(records=[{}, {}], input_record_count=1)


def test_manifest_comparison_identifies_material_changes() -> None:
    first = make_manifest()
    second = make_manifest(
        config_threshold=0.90,
        records=[{"message": "changed"}],
        pipeline_version="0.2.0",
        input_record_count=3,
    )

    comparison = compare_dataset_manifests(first, second)

    assert comparison.same_dataset_version is False
    assert comparison.input_changed is False
    assert comparison.config_changed is True
    assert comparison.output_changed is True
    assert comparison.pipeline_version_changed is True
    assert comparison.counts_changed is True


def test_manifest_write_and_load_round_trip(tmp_path) -> None:
    manifest = make_manifest()
    output_path = tmp_path / "nested" / "dataset_manifest.json"

    write_dataset_manifest(manifest, output_path)
    loaded = load_dataset_manifest(output_path)

    assert loaded == manifest
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_installed_pipeline_version_matches_project_metadata() -> None:
    assert get_pipeline_version() == "0.1.0"
