"""Persistence for final Phase 1 pipeline outputs."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ticket_pipeline.data_quality import data_quality_report_to_json
from ticket_pipeline.models import Ticket
from ticket_pipeline.pipeline import PipelineResult, RejectedRecord
from ticket_pipeline.versioning import dataset_manifest_to_json


ACCEPTED_RECORDS_FILENAME = "accepted.jsonl"
REJECTED_RECORDS_FILENAME = "rejected.jsonl"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATA_QUALITY_REPORT_FILENAME = "data_quality_report.json"
_FAILURE_ERROR_FIELDS = frozenset({"loc", "msg", "type", "url"})


class OutputPersistenceError(RuntimeError):
    """Fatal failure while writing Step 15 output artifacts."""


@dataclass(frozen=True, slots=True)
class OutputArtifacts:
    """Paths written by ``write_pipeline_outputs``."""

    accepted_path: Path
    rejected_path: Path
    manifest_path: Path
    report_path: Path


def write_pipeline_outputs(
    result: PipelineResult,
    output_dir: str | Path,
) -> OutputArtifacts:
    """Persist a completed ``PipelineResult`` to deterministic artifacts.

    The fixed artifact names are overwritten atomically. The function consumes
    the in-memory result as-is and does not rerun pipeline stages or mutate the
    supplied ``PipelineResult``.
    """

    target_dir = Path(output_dir)
    artifacts = OutputArtifacts(
        accepted_path=target_dir / ACCEPTED_RECORDS_FILENAME,
        rejected_path=target_dir / REJECTED_RECORDS_FILENAME,
        manifest_path=target_dir / DATASET_MANIFEST_FILENAME,
        report_path=target_dir / DATA_QUALITY_REPORT_FILENAME,
    )

    try:
        serialized_artifacts = {
            artifacts.accepted_path: serialize_accepted_records(
                result.accepted_records
            ),
            artifacts.rejected_path: serialize_rejected_records(
                result.rejected_records
            ),
            artifacts.manifest_path: (
                f"{dataset_manifest_to_json(result.dataset_manifest)}\n"
            ),
            artifacts.report_path: (
                f"{data_quality_report_to_json(result.data_quality_report)}\n"
            ),
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        for path, content in serialized_artifacts.items():
            _atomic_write_text(path, content)
    except OutputPersistenceError:
        raise
    except Exception as error:
        raise OutputPersistenceError(
            "Failed to persist pipeline output artifacts."
        ) from error

    return artifacts


def serialize_accepted_records(records: Sequence[Ticket]) -> str:
    """Serialize accepted processed tickets as deterministic JSONL."""

    return "".join(
        f"{_json_line(record.model_dump(mode='json'))}\n" for record in records
    )


def serialize_rejected_records(records: Sequence[RejectedRecord]) -> str:
    """Serialize rejected records as deterministic privacy-filtered JSONL."""

    return "".join(
        f"{_json_line(serialize_rejected_record(record))}\n"
        for record in records
    )


def serialize_rejected_record(record: RejectedRecord) -> dict[str, object]:
    """Return a privacy-safe rejected-record representation for disk output.

    Rejected records may fail before ever reaching PII masking, so no raw
    field value from the source or the canonical mapping is persisted -- only
    the names of the canonical fields that carried a value. ``record_index``
    plus sanitized ``failure`` metadata (diagnostic keys only, never Pydantic
    ``input``/``ctx``) locate and explain the rejection; the original values
    remain in the source file.
    """

    return {
        "record_index": record.record_index,
        "present_canonical_fields": _present_canonical_fields(
            record.canonical_record
        ),
        "failure": _safe_failure(record.failure),
    }


def _present_canonical_fields(record: Mapping[str, object]) -> list[str]:
    return sorted(
        str(key)
        for key, value in record.items()
        if isinstance(key, str) and _has_value(value)
    )


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return bool(value)
    return True


def _safe_failure(failure: Mapping[str, object]) -> dict[str, object]:
    reason = failure.get("reason")
    safe_failure: dict[str, object] = {
        "reason": reason if isinstance(reason, str) else "UNKNOWN"
    }
    errors = failure.get("errors")
    if isinstance(errors, list):
        safe_failure["errors"] = [
            _safe_validation_error(error)
            for error in errors
            if isinstance(error, Mapping)
        ]
    return safe_failure


def _safe_validation_error(error: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): _json_safe_value(value)
        for key, value in sorted(error.items())
        if isinstance(key, str) and key in _FAILURE_ERROR_FIELDS
    }


def _json_safe_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in sorted(value.items())
            if isinstance(key, str)
        }
    return value


def _json_line(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception as error:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise OutputPersistenceError(
            f"Failed to write output artifact: {path.name}"
        ) from error
