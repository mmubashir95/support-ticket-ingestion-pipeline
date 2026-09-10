import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from ticket_pipeline.data_quality import (
    generate_data_quality_report,
    load_data_quality_report,
)
from ticket_pipeline.language import LanguagePrediction
from ticket_pipeline.models import DatasetSourceManifest, SourceFileManifest, Ticket
from ticket_pipeline.outputs import (
    ACCEPTED_RECORDS_FILENAME,
    DATA_QUALITY_REPORT_FILENAME,
    DATASET_MANIFEST_FILENAME,
    REJECTED_RECORDS_FILENAME,
    OutputArtifacts,
    OutputPersistenceError,
    serialize_rejected_record,
    write_pipeline_outputs,
)
from ticket_pipeline.pipeline import (
    PipelineConfig,
    PipelineResult,
    RejectedRecord,
    run_pipeline,
)
from ticket_pipeline.validators import EMPTY_MESSAGE, SCHEMA_VALIDATION_FAILED
from ticket_pipeline.versioning import (
    capture_processing_config,
    create_dataset_manifest,
    load_dataset_manifest,
)


class DeterministicEmbeddingModel:
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        assert normalize_embeddings is True
        return np.asarray([[1.0, 0.0] for _ in sentences], dtype=np.float32)


class EnglishBackend:
    def predict(self, text: str) -> LanguagePrediction:
        return LanguagePrediction("en", 0.98)


def ticket(**updates: object) -> Ticket:
    values: dict[str, object] = {
        "subject": "Login issue",
        "message": "Cannot log in to my account.",
        "ticket_type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "source_version": 52,
        "tags": ["login"],
    }
    values.update(updates)
    return Ticket.model_validate(values)


def manifest(records: list[object], input_count: int):
    return create_dataset_manifest(
        source=DatasetSourceManifest(
            files=[
                SourceFileManifest(
                    name="tickets.csv",
                    sha256="a" * 64,
                    size_bytes=100,
                )
            ]
        ),
        processing_config=capture_processing_config(
            semantic_deduplication_enabled=True
        ),
        input_record_count=input_count,
        processed_records=records,
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )


def result_with(
    *,
    accepted: list[Ticket] | None = None,
    rejected: list[RejectedRecord] | None = None,
) -> PipelineResult:
    accepted_records = accepted or []
    rejected_records = rejected or []
    dataset_manifest = manifest(
        [record.model_dump(mode="python") for record in accepted_records],
        len(accepted_records) + len(rejected_records),
    )
    report = generate_data_quality_report(
        total_input_records=len(accepted_records) + len(rejected_records),
        accepted_records=accepted_records,
        dataset_manifest=dataset_manifest,
        validation_failures=[record.failure for record in rejected_records],
        exact_duplicate_results=[],
        semantic_duplicate_results=[],
        pii_results=[],
    )
    return PipelineResult(
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        dataset_manifest=dataset_manifest,
        data_quality_report=report,
    )


def rejected_record(
    *,
    record_index: int = 0,
    reason: str = EMPTY_MESSAGE,
) -> RejectedRecord:
    failure: dict[str, object] = {"reason": reason}
    if reason == SCHEMA_VALIDATION_FAILED:
        failure["errors"] = [
            {
                "loc": ("message",),
                "msg": "Input should be a valid string",
                "type": "string_type",
                "input": "raw-secret@example.com",
            }
        ]
    return RejectedRecord(
        record_index=record_index,
        canonical_record={
            "subject": "Contact Jane at jane@example.com",
            "message": "Call +1 555 123 4567",
            "ticket_type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "source_version": 52,
            "tags": ["login"],
        },
        failure=failure,  # type: ignore[arg-type]
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
        "version",
        "tag_1",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = [row.get(field, "") for field in fields]
        lines.append(",".join(f'"{value}"' for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def valid_row(**updates: str) -> dict[str, str]:
    row = {
        "subject": "Login issue",
        "body": "Cannot log in to my account.",
        "answer": "",
        "type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "version": "52",
        "tag_1": "login",
    }
    row.update(updates)
    return row


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if path.read_text(encoding="utf-8") == "":
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def artifact_contents(artifacts: OutputArtifacts) -> dict[str, str]:
    return {
        name: path.read_text(encoding="utf-8")
        for name, path in {
            "accepted": artifacts.accepted_path,
            "rejected": artifacts.rejected_path,
            "manifest": artifacts.manifest_path,
            "report": artifacts.report_path,
        }.items()
    }


def test_accepted_only_output_writes_records_and_empty_rejections(tmp_path) -> None:
    accepted = [
        ticket(subject="First", message="First message"),
        ticket(subject="Second", message="Second message"),
    ]
    result = result_with(accepted=accepted)

    artifacts = write_pipeline_outputs(result, tmp_path)

    assert isinstance(artifacts, OutputArtifacts)
    assert read_jsonl(artifacts.accepted_path) == [
        record.model_dump(mode="json") for record in accepted
    ]
    assert artifacts.rejected_path.read_text(encoding="utf-8") == ""
    assert load_dataset_manifest(artifacts.manifest_path) == result.dataset_manifest
    assert load_data_quality_report(artifacts.report_path) == (
        result.data_quality_report
    )


def test_rejected_only_output_writes_safe_rejections_and_empty_accepted(
    tmp_path,
) -> None:
    rejected = [rejected_record(record_index=7)]
    result = result_with(rejected=rejected)

    artifacts = write_pipeline_outputs(result, tmp_path)

    assert artifacts.accepted_path.read_text(encoding="utf-8") == ""
    rejected_rows = read_jsonl(artifacts.rejected_path)
    assert len(rejected_rows) == 1
    assert rejected_rows[0] == serialize_rejected_record(rejected[0])
    assert rejected_rows[0]["record_index"] == 7
    assert rejected_rows[0]["failure"] == {"reason": EMPTY_MESSAGE}


def test_mixed_output_preserves_counts_and_order(tmp_path) -> None:
    first = ticket(subject="First")
    second = ticket(subject="Second")
    rejected = [
        rejected_record(record_index=1),
        rejected_record(record_index=3, reason=SCHEMA_VALIDATION_FAILED),
    ]
    result = result_with(accepted=[first, second], rejected=rejected)

    artifacts = write_pipeline_outputs(result, tmp_path)

    accepted_rows = read_jsonl(artifacts.accepted_path)
    rejected_rows = read_jsonl(artifacts.rejected_path)
    assert [row["subject"] for row in accepted_rows] == ["First", "Second"]
    assert [row["record_index"] for row in rejected_rows] == [1, 3]
    assert len(accepted_rows) == len(result.accepted_records)
    assert len(rejected_rows) == len(result.rejected_records)


def test_completely_empty_result_writes_empty_jsonl_and_metadata(tmp_path) -> None:
    result = result_with()

    artifacts = write_pipeline_outputs(result, tmp_path)

    assert artifacts.accepted_path.read_text(encoding="utf-8") == ""
    assert artifacts.rejected_path.read_text(encoding="utf-8") == ""
    assert artifacts.manifest_path.exists()
    assert artifacts.report_path.exists()
    assert load_dataset_manifest(artifacts.manifest_path) == result.dataset_manifest
    assert load_data_quality_report(artifacts.report_path) == (
        result.data_quality_report
    )


def test_output_directory_is_created_with_stable_artifact_names(tmp_path) -> None:
    output_dir = tmp_path / "nested" / "outputs"
    result = result_with(accepted=[ticket()])

    artifacts = write_pipeline_outputs(result, output_dir)

    assert artifacts.accepted_path == output_dir / ACCEPTED_RECORDS_FILENAME
    assert artifacts.rejected_path == output_dir / REJECTED_RECORDS_FILENAME
    assert artifacts.manifest_path == output_dir / DATASET_MANIFEST_FILENAME
    assert artifacts.report_path == output_dir / DATA_QUALITY_REPORT_FILENAME
    assert output_dir.is_dir()


def test_existing_files_are_overwritten_deterministically(tmp_path) -> None:
    result = result_with(accepted=[ticket()])
    for name in (
        ACCEPTED_RECORDS_FILENAME,
        REJECTED_RECORDS_FILENAME,
        DATASET_MANIFEST_FILENAME,
        DATA_QUALITY_REPORT_FILENAME,
    ):
        (tmp_path / name).write_text("stale content", encoding="utf-8")

    artifacts = write_pipeline_outputs(result, tmp_path)

    assert "stale content" not in artifacts.accepted_path.read_text(
        encoding="utf-8"
    )
    assert read_jsonl(artifacts.accepted_path) == [
        result.accepted_records[0].model_dump(mode="json")
    ]
    assert load_dataset_manifest(artifacts.manifest_path) == result.dataset_manifest
    assert load_data_quality_report(artifacts.report_path) == (
        result.data_quality_report
    )


def test_deterministic_writes_produce_identical_contents(tmp_path) -> None:
    result = result_with(
        accepted=[ticket(subject="A"), ticket(subject="B")],
        rejected=[rejected_record(record_index=2)],
    )

    first = write_pipeline_outputs(result, tmp_path)
    first_contents = artifact_contents(first)
    second = write_pipeline_outputs(result, tmp_path)
    second_contents = artifact_contents(second)

    assert first_contents == second_contents


def test_privacy_safe_rejected_serialization_omits_raw_values_and_error_input() -> None:
    rejected = rejected_record(reason=SCHEMA_VALIDATION_FAILED)

    serialized = serialize_rejected_record(rejected)
    serialized_text = json.dumps(serialized, sort_keys=True)

    assert "canonical_record" not in serialized
    assert serialized["present_canonical_fields"] == [
        "language",
        "message",
        "priority",
        "queue",
        "source_version",
        "subject",
        "tags",
        "ticket_type",
    ]
    # No raw value from any canonical field or Pydantic error reaches disk.
    for secret in (
        "jane@example.com",
        "+1 555 123 4567",
        "raw-secret@example.com",
        "Technical Support",
        "login",
    ):
        assert secret not in serialized_text
    assert serialized["failure"] == {
        "reason": SCHEMA_VALIDATION_FAILED,
        "errors": [
            {
                "loc": ["message"],
                "msg": "Input should be a valid string",
                "type": "string_type",
            }
        ],
    }


def test_rejected_output_persists_no_raw_canonical_values(tmp_path) -> None:
    rejected = [
        RejectedRecord(
            record_index=0,
            canonical_record={
                "subject": "Reset for john.doe@corp.com",
                "message": "card 4111 1111 1111 1111 was charged",
                "ticket_type": "Incident",
                "queue": "escalate to alice@corp.com",
                "priority": "critical-not-valid",
                "language": "en",
                "source_version": "v-4111111111111111",
                "tags": ["ssn 123-45-6789"],
            },
            failure={  # type: ignore[arg-type]
                "reason": SCHEMA_VALIDATION_FAILED,
                "errors": [
                    {
                        "loc": ("priority",),
                        "msg": "Input should be 'low', 'medium' or 'high'",
                        "type": "literal_error",
                        "input": "critical-not-valid",
                        "ctx": {"expected": "'low', 'medium' or 'high'"},
                    }
                ],
            },
        )
    ]
    result = result_with(rejected=rejected)

    artifacts = write_pipeline_outputs(result, tmp_path)
    raw = artifacts.rejected_path.read_text(encoding="utf-8")

    for secret in (
        "john.doe@corp.com",
        "alice@corp.com",
        "4111",
        "123-45-6789",
        "critical-not-valid",
        "v-4111111111111111",
        "Incident",
        "escalate",
    ):
        assert secret not in raw

    row = read_jsonl(artifacts.rejected_path)[0]
    assert "canonical_record" not in row
    assert row["present_canonical_fields"] == [
        "language",
        "message",
        "priority",
        "queue",
        "source_version",
        "subject",
        "tags",
        "ticket_type",
    ]
    assert row["failure"]["errors"][0] == {
        "loc": ["priority"],
        "msg": "Input should be 'low', 'medium' or 'high'",
        "type": "literal_error",
    }


def test_atomic_write_failure_cleans_up_temp_file_and_raises(
    tmp_path, monkeypatch
) -> None:
    import ticket_pipeline.outputs as outputs_module

    def failing_fsync(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(outputs_module.os, "fsync", failing_fsync)

    with pytest.raises(OutputPersistenceError) as error:
        write_pipeline_outputs(result_with(accepted=[ticket()]), tmp_path)

    assert isinstance(error.value.__cause__, OSError)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_original_pipeline_result_is_not_mutated_by_persistence(tmp_path) -> None:
    result = result_with(
        accepted=[ticket(subject="Stable")],
        rejected=[rejected_record(reason=SCHEMA_VALIDATION_FAILED)],
    )
    snapshot = replace(
        result,
        accepted_records=list(result.accepted_records),
        rejected_records=list(result.rejected_records),
    )

    write_pipeline_outputs(result, tmp_path)

    assert result == snapshot


def test_invalid_output_path_surfaces_persistence_error(tmp_path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("content", encoding="utf-8")

    with pytest.raises(OutputPersistenceError) as error:
        write_pipeline_outputs(result_with(accepted=[ticket()]), blocking_file)

    assert isinstance(error.value.__cause__, FileExistsError)


def test_step14_to_step15_integration_persists_result_artifacts(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [
            valid_row(body="Call +1 555 123 4567."),
            valid_row(body="   "),
            valid_row(answer="Agent response"),
        ],
    )
    result = run_pipeline(
        source,
        PipelineConfig(
            semantic_model=DeterministicEmbeddingModel(),
            semantic_model_name="test-output-embedding-v1",
            language_backend=EnglishBackend(),
            pipeline_version="0.1.0",
            created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        ),
    )

    artifacts = write_pipeline_outputs(result, tmp_path / "outputs")

    assert read_jsonl(artifacts.accepted_path) == [
        record.model_dump(mode="json") for record in result.accepted_records
    ]
    assert read_jsonl(artifacts.rejected_path) == [
        serialize_rejected_record(record) for record in result.rejected_records
    ]
    assert load_dataset_manifest(artifacts.manifest_path) == result.dataset_manifest
    assert load_data_quality_report(artifacts.report_path) == (
        result.data_quality_report
    )
    assert result.accepted_records[0].message == "Call <PHONE>."
    assert result.rejected_records[0].failure["reason"] == EMPTY_MESSAGE
