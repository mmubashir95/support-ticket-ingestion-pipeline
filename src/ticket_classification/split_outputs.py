"""Persistence for frozen classification dataset split artifacts."""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from ticket_classification.models import ClassificationRecord
from ticket_classification.splitting import (
    SplitConfig,
    SplitIntegrityReport,
    SplitResult,
    validate_split_integrity,
)


TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"
TEST_FILENAME = "test.jsonl"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"


class SplitArtifactPersistenceError(RuntimeError):
    """A frozen split artifact could not be written or verified."""


class SplitPolicy(BaseModel):
    """Human-readable description of how split membership was assigned."""

    model_config = ConfigDict(extra="forbid")

    allocation: Literal["floor/floor/remainder"] = "floor/floor/remainder"
    stratification_key: Literal["label"] | None = None


class SplitManifest(BaseModel):
    """Deterministic metadata and exact membership of one frozen split."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr
    target: StrictStr
    task_type: Literal["single_label_multiclass"] = "single_label_multiclass"
    random_seed: StrictInt
    split_policy: SplitPolicy
    train_ratio: float = Field(gt=0.0, lt=1.0)
    validation_ratio: float = Field(gt=0.0, lt=1.0)
    test_ratio: float = Field(gt=0.0, lt=1.0)
    stratified: StrictBool
    total_count: int = Field(ge=1)
    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    class_counts_full: dict[StrictStr, int]
    class_counts_train: dict[StrictStr, int]
    class_counts_validation: dict[StrictStr, int]
    class_counts_test: dict[StrictStr, int]
    train_record_ids: list[StrictStr]
    validation_record_ids: list[StrictStr]
    test_record_ids: list[StrictStr]

    @model_validator(mode="after")
    def validate_membership_counts(self) -> "SplitManifest":
        memberships = (
            ("train", self.train_count, self.train_record_ids),
            ("validation", self.validation_count, self.validation_record_ids),
            ("test", self.test_count, self.test_record_ids),
        )
        for name, count, record_ids in memberships:
            if len(record_ids) != count:
                raise ValueError(
                    f"{name}_record_ids length must equal {name}_count"
                )
        all_ids = [record_id for _, _, ids in memberships for record_id in ids]
        if len(all_ids) != self.total_count:
            raise ValueError("split record ID count must equal total_count")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("split manifest record IDs must be unique")
        if self.train_count + self.validation_count + self.test_count != self.total_count:
            raise ValueError("split counts must add up to total_count")
        return self


@dataclass(frozen=True, slots=True)
class SplitArtifacts:
    """Paths for one persisted and verified frozen split."""

    train_path: Path
    validation_path: Path
    test_path: Path
    manifest_path: Path


def write_split_artifacts(
    records: Sequence[ClassificationRecord],
    result: SplitResult,
    config: SplitConfig,
    output_dir: str | Path,
    *,
    expected_dataset_version: str,
) -> SplitArtifacts:
    """Validate, atomically write, and read back one frozen dataset split."""

    report = validate_split_integrity(
        records,
        config,
        result,
        expected_dataset_version=expected_dataset_version,
    )
    manifest = build_split_manifest(result, config, report)
    target_dir = Path(output_dir)
    artifacts = SplitArtifacts(
        train_path=target_dir / TRAIN_FILENAME,
        validation_path=target_dir / VALIDATION_FILENAME,
        test_path=target_dir / TEST_FILENAME,
        manifest_path=target_dir / SPLIT_MANIFEST_FILENAME,
    )

    try:
        serialized_artifacts = {
            artifacts.train_path: serialize_classification_records(result.train),
            artifacts.validation_path: serialize_classification_records(
                result.validation
            ),
            artifacts.test_path: serialize_classification_records(result.test),
            artifacts.manifest_path: f"{split_manifest_to_json(manifest)}\n",
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        for path, content in serialized_artifacts.items():
            _atomic_write_text(path, content)
        _verify_written_artifacts(
            records,
            result,
            config,
            manifest,
            artifacts,
            expected_dataset_version=expected_dataset_version,
        )
    except SplitArtifactPersistenceError:
        raise
    except Exception as error:
        raise SplitArtifactPersistenceError(
            "Failed to persist and verify split artifacts."
        ) from error

    return artifacts


def build_split_manifest(
    result: SplitResult,
    config: SplitConfig,
    report: SplitIntegrityReport,
) -> SplitManifest:
    """Construct deterministic metadata for an already validated split."""

    return SplitManifest(
        dataset_version=report.dataset_version,
        target=config.target_field,
        random_seed=config.random_seed,
        split_policy=SplitPolicy(
            stratification_key="label" if config.stratify else None
        ),
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
        stratified=config.stratify,
        total_count=report.expected_total,
        train_count=report.train_count,
        validation_count=report.validation_count,
        test_count=report.test_count,
        class_counts_full=report.class_counts["full"],
        class_counts_train=report.class_counts["train"],
        class_counts_validation=report.class_counts["validation"],
        class_counts_test=report.class_counts["test"],
        train_record_ids=[record.record_id for record in result.train],
        validation_record_ids=[record.record_id for record in result.validation],
        test_record_ids=[record.record_id for record in result.test],
    )


def serialize_classification_records(
    records: Sequence[ClassificationRecord],
) -> str:
    """Serialize classification records as deterministic JSONL."""

    return "".join(
        f"{_json_line(record.model_dump(mode='json'))}\n" for record in records
    )


def split_manifest_to_json(manifest: SplitManifest) -> str:
    """Serialize a split manifest as deterministic readable JSON."""

    return json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )


def load_classification_records(path: str | Path) -> list[ClassificationRecord]:
    """Read and validate a frozen classification JSONL artifact."""

    loaded: list[ClassificationRecord] = []
    with Path(path).open(encoding="utf-8") as artifact_file:
        for line_number, line in enumerate(artifact_file, start=1):
            if not line.strip():
                continue
            try:
                loaded.append(ClassificationRecord.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as error:
                raise SplitArtifactPersistenceError(
                    f"Invalid classification JSONL at line {line_number}: "
                    f"{Path(path).name}"
                ) from error
    return loaded


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Read and validate a frozen split manifest."""

    try:
        return SplitManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SplitArtifactPersistenceError(
            f"Invalid split manifest: {Path(path).name}"
        ) from error


def _verify_written_artifacts(
    records: Sequence[ClassificationRecord],
    result: SplitResult,
    config: SplitConfig,
    manifest: SplitManifest,
    artifacts: SplitArtifacts,
    *,
    expected_dataset_version: str,
) -> None:
    loaded_result = SplitResult(
        train=load_classification_records(artifacts.train_path),
        validation=load_classification_records(artifacts.validation_path),
        test=load_classification_records(artifacts.test_path),
    )
    loaded_manifest = load_split_manifest(artifacts.manifest_path)
    if loaded_result != result:
        raise SplitArtifactPersistenceError(
            "Read-back split records do not match the in-memory split."
        )
    if loaded_manifest != manifest:
        raise SplitArtifactPersistenceError(
            "Read-back split manifest does not match the generated manifest."
        )

    validate_split_integrity(
        records,
        config,
        loaded_result,
        expected_dataset_version=expected_dataset_version,
    )
    manifest_ids = (
        loaded_manifest.train_record_ids,
        loaded_manifest.validation_record_ids,
        loaded_manifest.test_record_ids,
    )
    artifact_ids = tuple(
        [record.record_id for record in group]
        for group in (
            loaded_result.train,
            loaded_result.validation,
            loaded_result.test,
        )
    )
    if manifest_ids != artifact_ids:
        raise SplitArtifactPersistenceError(
            "Manifest record IDs do not match persisted JSONL order."
        )


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
        raise SplitArtifactPersistenceError(
            f"Failed to write split artifact: {path.name}"
        ) from error
