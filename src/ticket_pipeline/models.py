"""Canonical data models for support tickets.

This module defines structural schema validity only. Business checks such as
empty-message detection, HTML cleanup, PII masking, and duplicate detection are
handled by later pipeline phases.
"""

import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


Priority = Literal["low", "medium", "high"]


class LanguageDetectionStatus(str, Enum):
    """Outcome of generated ticket-language detection."""

    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class LanguageDetectionMetadata(BaseModel):
    """Validated metadata produced by the language-detection stage."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: LanguageDetectionStatus

    @model_validator(mode="after")
    def validate_status_fields(self) -> "LanguageDetectionMetadata":
        """Keep language and confidence consistent with the outcome status."""

        if self.status is LanguageDetectionStatus.DETECTED:
            if self.language is None or self.confidence is None:
                raise ValueError("detected status requires language and confidence")
        elif self.language is not None:
            raise ValueError("non-detected status must not include a language")

        if (
            self.status is LanguageDetectionStatus.FAILED
            and self.confidence is not None
        ):
            raise ValueError("failed status must not include confidence")
        return self


class LeakageCheckStatus(str, Enum):
    """Outcome of policy-based leakage inspection."""

    CLEAN = "clean"
    WARNING = "warning"
    FAILED = "failed"


class LeakageIssueType(str, Enum):
    """Configured feature risks reported by the leakage checker.

    Only genuinely risky, actionable findings belong here. A target field
    being present is not one of them: a labeled training row is expected to
    contain its own label, so target/proxy presence is recorded separately as
    informational ``LeakageTargetMetadata``, not as a warning-worthy issue.
    """

    FUTURE_FIELD = "future_field"


class LeakageIssue(BaseModel):
    """One populated field that violates the configured feature policy."""

    model_config = ConfigDict(extra="forbid")

    type: LeakageIssueType
    field: StrictStr


GroupIdentifierValue = StrictStr | StrictInt | StrictFloat | StrictBool


class LeakageGroupingMetadata(BaseModel):
    """Information that future dataset splitting must keep together."""

    model_config = ConfigDict(extra="forbid")

    exact_duplicate_group_id: int | None = Field(default=None, ge=0)
    semantic_duplicate_group_id: int | None = Field(default=None, ge=0)
    identifiers: dict[StrictStr, GroupIdentifierValue] = Field(default_factory=dict)


class LeakageTargetMetadata(BaseModel):
    """Configured target/proxy fields populated on this record.

    This is informational, like ``LeakageGroupingMetadata``: a populated
    target field is the normal, expected shape of a labeled training record,
    not evidence that the record itself is invalid. It exists so whoever
    builds the input-feature set for the named target task knows which
    fields to exclude from it.
    """

    model_config = ConfigDict(extra="forbid")

    target_fields: list[StrictStr] = Field(default_factory=list)
    target_proxy_fields: list[StrictStr] = Field(default_factory=list)


class LeakageCheckMetadata(BaseModel):
    """Structured leakage warnings and informational grouping metadata."""

    model_config = ConfigDict(extra="forbid")

    status: LeakageCheckStatus
    issues: list[LeakageIssue] = Field(default_factory=list)
    targets: LeakageTargetMetadata = Field(default_factory=LeakageTargetMetadata)
    grouping: LeakageGroupingMetadata = Field(
        default_factory=LeakageGroupingMetadata
    )
    unchecked_fields: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> "LeakageCheckMetadata":
        """Require warning status exactly when policy issues are present."""

        if self.status is LeakageCheckStatus.WARNING and not self.issues:
            raise ValueError("warning status requires at least one issue")
        if self.status is not LeakageCheckStatus.WARNING and self.issues:
            raise ValueError("leakage issues require warning status")
        return self


class Ticket(BaseModel):
    """Canonical representation of one support ticket."""

    model_config = ConfigDict(extra="forbid")

    subject: StrictStr | None = None
    message: StrictStr
    ticket_type: StrictStr
    queue: StrictStr
    priority: Priority
    language: StrictStr
    source_version: int
    tags: list[StrictStr] = Field(default_factory=list)
    language_detection: LanguageDetectionMetadata | None = None
    leakage_check: LeakageCheckMetadata | None = None


class SourceFileManifest(BaseModel):
    """Stable identity for one file used to build a dataset."""

    model_config = ConfigDict(extra="forbid")

    name: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if not normalized.strip() or "/" in normalized or "\\" in normalized:
            raise ValueError("source name must be a non-blank file name")
        return normalized


class DatasetSourceManifest(BaseModel):
    """Deterministically ordered source-file identities."""

    model_config = ConfigDict(extra="forbid")

    files: list[SourceFileManifest] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def sort_and_validate_files(
        cls,
        files: list[SourceFileManifest],
    ) -> list[SourceFileManifest]:
        ordered = sorted(files, key=lambda item: item.name)
        names = [item.name for item in ordered]
        if len(names) != len(set(names)):
            raise ValueError("source file names must be unique")
        return ordered


class DatasetRecordCounts(BaseModel):
    """Counts available before accepted/rejected outputs are implemented."""

    model_config = ConfigDict(extra="forbid")

    input: int = Field(ge=0)
    processed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "DatasetRecordCounts":
        if self.processed > self.input:
            raise ValueError("processed count cannot exceed input count")
        return self


class DatasetFingerprints(BaseModel):
    """SHA-256 identities for dataset inputs, configuration, and output."""

    model_config = ConfigDict(extra="forbid")

    input: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    config: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    output: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")


class SemanticDeduplicationConfigSnapshot(BaseModel):
    """Material semantic-deduplication settings for one dataset build."""

    model_config = ConfigDict(extra="forbid")

    model_name: StrictStr = Field(min_length=1)
    threshold: float = Field(ge=-1.0, le=1.0)
    top_k: int = Field(ge=1)
    batch_size: int = Field(ge=1)


class LanguageDetectionConfigSnapshot(BaseModel):
    """Material language-detection settings for one dataset build."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["lingua-language-detector"]
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    min_alphabetic_characters: int = Field(ge=1)
    supported_languages: list[StrictStr] = Field(min_length=1)


class LeakageConfigSnapshot(BaseModel):
    """Deterministically ordered leakage policy for one dataset build."""

    model_config = ConfigDict(extra="forbid")

    forbidden_fields: list[StrictStr] = Field(default_factory=list)
    target_fields: list[StrictStr] = Field(default_factory=list)
    target_proxy_fields: list[StrictStr] = Field(default_factory=list)
    group_fields: list[StrictStr] = Field(default_factory=list)


class ProcessingConfigSnapshot(BaseModel):
    """Existing configurable settings that materially affect processing."""

    model_config = ConfigDict(extra="forbid")

    semantic_deduplication: SemanticDeduplicationConfigSnapshot | None
    language_detection: LanguageDetectionConfigSnapshot
    leakage: LeakageConfigSnapshot


class DatasetManifest(BaseModel):
    """Content-addressed provenance for one processed dataset build."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr = Field(pattern=r"^ds_[a-f0-9]{12}$")
    created_at: datetime
    pipeline_version: StrictStr = Field(min_length=1)
    source: DatasetSourceManifest
    counts: DatasetRecordCounts
    processing_config: ProcessingConfigSnapshot
    fingerprints: DatasetFingerprints

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)


class DatasetManifestComparison(BaseModel):
    """Material differences between two dataset manifests."""

    model_config = ConfigDict(extra="forbid")

    same_dataset_version: bool
    input_changed: bool
    config_changed: bool
    output_changed: bool
    pipeline_version_changed: bool
    counts_changed: bool


class DatasetSummary(BaseModel):
    """Accepted and rejected record counts for one report input."""

    model_config = ConfigDict(extra="forbid")

    total_input_records: int = Field(ge=0)
    accepted_records: int = Field(ge=0)
    rejected_records: int = Field(ge=0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_totals(self) -> "DatasetSummary":
        if self.accepted_records + self.rejected_records != self.total_input_records:
            raise ValueError(
                "accepted and rejected records must equal total input records"
            )
        return self


class ValidationSummary(BaseModel):
    """Aggregated validation failures from existing rejection metadata."""

    model_config = ConfigDict(extra="forbid")

    records_with_validation_issues: int = Field(ge=0)
    validation_issue_events: int = Field(ge=0)
    issue_counts_by_reason: dict[StrictStr, int] = Field(default_factory=dict)
    schema_error_counts_by_type: dict[StrictStr, int] = Field(default_factory=dict)
    schema_error_counts_by_field: dict[StrictStr, int] = Field(default_factory=dict)


class PreprocessingSummary(BaseModel):
    """Transformation metrics supplied by previous preprocessing stages."""

    model_config = ConfigDict(extra="forbid")

    metrics: dict[StrictStr, int] = Field(default_factory=dict)
    limitations: list[StrictStr] = Field(default_factory=list)


class PiiSummary(BaseModel):
    """PII masking counts only; raw PII values are never retained."""

    model_config = ConfigDict(extra="forbid")

    records_with_pii: int = Field(ge=0)
    total_entities_masked: int = Field(ge=0)
    entities_by_type: dict[StrictStr, int] = Field(default_factory=dict)
    limitations: list[StrictStr] = Field(default_factory=list)


class ExactDeduplicationSummary(BaseModel):
    """Counts derived from Step 8 exact duplicate results."""

    model_config = ConfigDict(extra="forbid")

    records_evaluated: int = Field(ge=0)
    duplicate_records_detected: int = Field(ge=0)
    duplicate_groups: int = Field(ge=0)


class SemanticDeduplicationSummary(BaseModel):
    """Counts derived from Step 9 semantic duplicate candidate results."""

    model_config = ConfigDict(extra="forbid")

    records_evaluated: int = Field(ge=0)
    duplicate_candidate_records_detected: int = Field(ge=0)
    duplicate_candidate_groups: int = Field(ge=0)
    skipped_exact_duplicate_records: int = Field(ge=0)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class DeduplicationSummary(BaseModel):
    """Exact and semantic deduplication aggregates."""

    model_config = ConfigDict(extra="forbid")

    exact: ExactDeduplicationSummary
    semantic: SemanticDeduplicationSummary


class LanguageSummary(BaseModel):
    """Aggregated Step 10 language-detection metadata."""

    model_config = ConfigDict(extra="forbid")

    checks_performed: int = Field(ge=0)
    detected_records: int = Field(ge=0)
    unknown_records: int = Field(ge=0)
    low_confidence_records: int = Field(ge=0)
    failed_records: int = Field(ge=0)
    by_language: dict[StrictStr, int] = Field(default_factory=dict)
    percentages_by_language: dict[StrictStr, float] = Field(default_factory=dict)


class LeakageSummary(BaseModel):
    """Aggregated Step 11 leakage-check metadata without raw leaked content."""

    model_config = ConfigDict(extra="forbid")

    checks_performed: int = Field(ge=0)
    clean_records: int = Field(ge=0)
    warning_records: int = Field(ge=0)
    failed_records: int = Field(ge=0)
    violations_detected: int = Field(ge=0)
    warnings_detected: int = Field(ge=0)
    issues_by_type: dict[StrictStr, int] = Field(default_factory=dict)
    issues_by_field: dict[StrictStr, int] = Field(default_factory=dict)


class DatasetVersionSummary(BaseModel):
    """Authoritative Step 12 identity copied from the dataset manifest."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: StrictStr = Field(pattern=r"^ds_[a-f0-9]{12}$")
    pipeline_version: StrictStr = Field(min_length=1)
    input_fingerprint: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    config_fingerprint: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    output_fingerprint: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")


class DataQualityReport(BaseModel):
    """Deterministic data-quality report for one processed dataset version."""

    model_config = ConfigDict(extra="forbid")

    dataset_summary: DatasetSummary
    validation_summary: ValidationSummary
    preprocessing_summary: PreprocessingSummary
    pii_summary: PiiSummary
    deduplication_summary: DeduplicationSummary
    language_summary: LanguageSummary
    leakage_summary: LeakageSummary
    dataset_version: DatasetVersionSummary
