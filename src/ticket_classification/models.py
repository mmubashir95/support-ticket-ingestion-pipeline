"""Typed contracts for classification dataset readiness audits."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


ClassificationTaskType = Literal[
    "binary",
    "single_label_multiclass",
    "multilabel",
    "not_supervised",
]


class FieldUsage(str, Enum):
    """How a ticket field may be used by future classification pipelines."""

    ALLOWED_INPUT = "ALLOWED_INPUT"
    TARGET = "TARGET"
    IDENTIFIER = "IDENTIFIER"
    METADATA_ONLY = "METADATA_ONLY"
    LEAKAGE_RISK = "LEAKAGE_RISK"
    NOT_AVAILABLE_AT_INFERENCE = "NOT_AVAILABLE_AT_INFERENCE"


class DatasetReadinessStatus(str, Enum):
    """Production readiness for supervised classification dataset use."""

    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY = "NOT_READY"


class ClassificationDatasetConfigSnapshot(BaseModel):
    """Deterministic configuration that materially affects the audit."""

    model_config = ConfigDict(extra="forbid")

    target_field: StrictStr
    input_fields: list[StrictStr] = Field(min_length=1)
    rare_class_min_samples: int = Field(ge=1)
    short_text_min_words: int = Field(ge=1)
    placeholder_only_tokens: list[StrictStr] = Field(default_factory=list)


class ClassificationRecord(BaseModel):
    """One record representation suitable for future classifier training."""

    model_config = ConfigDict(extra="forbid")

    record_id: StrictStr
    text: StrictStr
    label: StrictStr | list[StrictStr]
    source_fields: list[StrictStr] = Field(min_length=1)


class ClassificationDatasetSummary(BaseModel):
    """High-level counts for accepted Phase 1 records."""

    model_config = ConfigDict(extra="forbid")

    total_records: int = Field(ge=0)
    usable_labeled_records: int = Field(ge=0)
    dataset_version: StrictStr | None = None
    input_fingerprint: StrictStr | None = None
    output_fingerprint: StrictStr | None = None


class ClassificationTaskSummary(BaseModel):
    """Detected supervised task shape."""

    model_config = ConfigDict(extra="forbid")

    target_field: StrictStr
    task_type: ClassificationTaskType
    input_fields: list[StrictStr] = Field(min_length=1)
    source_fields_used_to_construct_input: list[StrictStr] = Field(min_length=1)


class TargetSummary(BaseModel):
    """Target-label completeness and label-quality summary."""

    model_config = ConfigDict(extra="forbid")

    total_records: int = Field(ge=0)
    records_with_usable_labels: int = Field(ge=0)
    records_with_missing_labels: int = Field(ge=0)
    records_with_blank_labels: int = Field(ge=0)
    records_with_invalid_labels: int = Field(ge=0)
    unique_class_count: int = Field(ge=0)
    class_names: list[StrictStr] = Field(default_factory=list)
    suspicious_label_groups: list[list[StrictStr]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "TargetSummary":
        checked = (
            self.records_with_usable_labels
            + self.records_with_missing_labels
            + self.records_with_blank_labels
            + self.records_with_invalid_labels
        )
        if checked != self.total_records:
            raise ValueError("target counts must add up to total records")
        return self


class ClassDistributionItem(BaseModel):
    """One class frequency row."""

    model_config = ConfigDict(extra="forbid")

    class_name: StrictStr
    count: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=1.0)


class ClassDistributionSummary(BaseModel):
    """Class distribution, imbalance, and rare-class metadata."""

    model_config = ConfigDict(extra="forbid")

    classes: list[ClassDistributionItem] = Field(default_factory=list)
    largest_class: StrictStr | None = None
    smallest_class: StrictStr | None = None
    largest_to_smallest_ratio: float | None = Field(default=None, ge=0.0)
    rare_class_min_samples: int = Field(ge=1)
    rare_classes: list[ClassDistributionItem] = Field(default_factory=list)
    imbalance_severity: StrictStr


class MissingTargetSummary(BaseModel):
    """Explicit target state counts."""

    model_config = ConfigDict(extra="forbid")

    valid_target_records: int = Field(ge=0)
    missing_target_records: int = Field(ge=0)
    blank_target_records: int = Field(ge=0)
    invalid_target_records: int = Field(ge=0)


class InputFieldPolicyItem(BaseModel):
    """Classification-specific field usage decision."""

    model_config = ConfigDict(extra="forbid")

    field_name: StrictStr
    usage: FieldUsage
    reason: StrictStr


class TextLengthStatistics(BaseModel):
    """Stable descriptive statistics for text lengths."""

    model_config = ConfigDict(extra="forbid")

    minimum: int = Field(ge=0)
    mean: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    p90: int = Field(ge=0)
    p95: int = Field(ge=0)
    p99: int = Field(ge=0)
    maximum: int = Field(ge=0)


class TextLengthSummary(BaseModel):
    """Character and token-like word length distribution."""

    model_config = ConfigDict(extra="forbid")

    character_length: TextLengthStatistics
    word_length: TextLengthStatistics
    empty_input_records: int = Field(ge=0)
    short_input_records: int = Field(ge=0)
    placeholder_only_records: int = Field(ge=0)
    short_text_min_words: int = Field(ge=1)


class DuplicateInputSummary(BaseModel):
    """Exact classification-input duplicate and conflict summary."""

    model_config = ConfigDict(extra="forbid")

    duplicated_input_groups: int = Field(ge=0)
    groups_with_consistent_labels: int = Field(ge=0)
    groups_with_conflicting_labels: int = Field(ge=0)
    conflicting_examples: list[dict[str, object]] = Field(default_factory=list)


class LeakageRiskItem(BaseModel):
    """Classification-specific feature leakage concern."""

    model_config = ConfigDict(extra="forbid")

    field_name: StrictStr
    reason: StrictStr


class ClassificationDatasetAudit(BaseModel):
    """Complete structured classification-readiness audit result."""

    model_config = ConfigDict(extra="forbid")

    audit_config: ClassificationDatasetConfigSnapshot
    dataset_summary: ClassificationDatasetSummary
    classification_task: ClassificationTaskSummary
    target_summary: TargetSummary
    class_distribution: ClassDistributionSummary
    rare_classes: list[ClassDistributionItem] = Field(default_factory=list)
    input_field_policy: list[InputFieldPolicyItem] = Field(default_factory=list)
    text_length_summary: TextLengthSummary
    missing_target_summary: MissingTargetSummary
    conflicting_labels: DuplicateInputSummary
    leakage_risks: list[LeakageRiskItem] = Field(default_factory=list)
    warnings: list[StrictStr] = Field(default_factory=list)
    dataset_readiness: DatasetReadinessStatus

