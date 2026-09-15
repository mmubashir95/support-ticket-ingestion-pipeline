"""Production-readiness audit for supervised ticket classification data."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from ticket_pipeline.models import DatasetManifest, Ticket

from ticket_classification.dataset import (
    ClassificationAuditConfig,
    build_classification_text,
    build_record_ids,
    target_value,
)
from ticket_classification.models import (
    ClassDistributionItem,
    ClassDistributionSummary,
    ClassificationDatasetAudit,
    ClassificationDatasetConfigSnapshot,
    ClassificationDatasetSummary,
    ClassificationTaskSummary,
    DatasetReadinessStatus,
    DuplicateInputSummary,
    FieldUsage,
    InputFieldPolicyItem,
    LeakageRiskItem,
    MissingTargetSummary,
    TargetSummary,
    TextLengthStatistics,
    TextLengthSummary,
)


RATE_DECIMAL_PLACES = 6
_WORD_PATTERN = re.compile(r"\S+")


def audit_classification_dataset(
    tickets: Sequence[Ticket],
    config: ClassificationAuditConfig | None = None,
    *,
    dataset_manifest: DatasetManifest | None = None,
) -> ClassificationDatasetAudit:
    """Audit accepted Phase 1 records for supervised classification readiness."""

    settings = config or ClassificationAuditConfig()
    ticket_list = list(tickets)
    _validate_fields(settings)

    label_states = [_classify_label(target_value(ticket, settings.target_field)) for ticket in ticket_list]
    usable_labels = [state[1] for state in label_states if state[0] == "valid"]
    class_counts = Counter(_iter_label_names(usable_labels))
    task_type = _detect_task_type(usable_labels)
    class_items = _class_distribution_items(class_counts, len(usable_labels))
    rare_classes = [
        item for item in class_items if item.count < settings.rare_class_min_samples
    ]
    suspicious_labels = _suspicious_label_groups(class_counts.keys())
    missing_summary = MissingTargetSummary(
        valid_target_records=sum(1 for state, _ in label_states if state == "valid"),
        missing_target_records=sum(1 for state, _ in label_states if state == "missing"),
        blank_target_records=sum(1 for state, _ in label_states if state == "blank"),
        invalid_target_records=sum(1 for state, _ in label_states if state == "invalid"),
    )
    target_summary = TargetSummary(
        total_records=len(ticket_list),
        records_with_usable_labels=missing_summary.valid_target_records,
        records_with_missing_labels=missing_summary.missing_target_records,
        records_with_blank_labels=missing_summary.blank_target_records,
        records_with_invalid_labels=missing_summary.invalid_target_records,
        unique_class_count=len(class_counts),
        class_names=sorted(class_counts),
        suspicious_label_groups=suspicious_labels,
    )
    input_texts = [
        build_classification_text(ticket, settings.input_fields)
        for ticket in ticket_list
    ]
    record_ids = build_record_ids(ticket_list)
    text_length_summary = _summarize_text_lengths(input_texts, settings)
    conflicting_labels = _duplicate_input_summary(
        input_texts,
        [state[1] for state in label_states],
        record_ids,
    )
    input_field_policy = _input_field_policy(settings)
    leakage_risks = _classification_leakage_risks(input_field_policy)
    warnings = _warnings(
        target_summary=target_summary,
        distribution=class_items,
        rare_classes=rare_classes,
        text_length_summary=text_length_summary,
        conflicting_labels=conflicting_labels,
    )
    readiness = _readiness_status(
        task_type=task_type,
        target_summary=target_summary,
        warnings=warnings,
    )
    manifest_version = dataset_manifest.dataset_version if dataset_manifest else None
    manifest_input = dataset_manifest.fingerprints.input if dataset_manifest else None
    manifest_output = dataset_manifest.fingerprints.output if dataset_manifest else None

    return ClassificationDatasetAudit(
        audit_config=ClassificationDatasetConfigSnapshot(
            target_field=settings.target_field,
            input_fields=list(settings.input_fields),
            rare_class_min_samples=settings.rare_class_min_samples,
            short_text_min_words=settings.short_text_min_words,
            placeholder_only_tokens=sorted(settings.placeholder_only_tokens),
        ),
        dataset_summary=ClassificationDatasetSummary(
            total_records=len(ticket_list),
            usable_labeled_records=target_summary.records_with_usable_labels,
            dataset_version=manifest_version,
            input_fingerprint=manifest_input,
            output_fingerprint=manifest_output,
        ),
        classification_task=ClassificationTaskSummary(
            target_field=settings.target_field,
            task_type=task_type,
            input_fields=list(settings.input_fields),
            source_fields_used_to_construct_input=list(settings.input_fields),
        ),
        target_summary=target_summary,
        class_distribution=ClassDistributionSummary(
            classes=class_items,
            largest_class=class_items[0].class_name if class_items else None,
            smallest_class=class_items[-1].class_name if class_items else None,
            largest_to_smallest_ratio=_largest_smallest_ratio(class_items),
            rare_class_min_samples=settings.rare_class_min_samples,
            rare_classes=rare_classes,
            imbalance_severity=_imbalance_severity(class_items),
        ),
        rare_classes=rare_classes,
        input_field_policy=input_field_policy,
        text_length_summary=text_length_summary,
        missing_target_summary=missing_summary,
        conflicting_labels=conflicting_labels,
        leakage_risks=leakage_risks,
        warnings=warnings,
        dataset_readiness=readiness,
    )


def _validate_fields(settings: ClassificationAuditConfig) -> None:
    ticket_fields = set(Ticket.model_fields)
    requested = set(settings.input_fields) | {settings.target_field}
    unknown = sorted(requested - ticket_fields)
    if unknown:
        raise ValueError(f"unknown Ticket field(s): {', '.join(unknown)}")
    for input_field in settings.input_fields:
        if input_field not in {"subject", "message"}:
            raise ValueError(f"unsupported classification input field: {input_field}")


def _classify_label(value: object) -> tuple[str, str | list[str] | None]:
    if value is None:
        return "missing", None
    if isinstance(value, str):
        if value.strip() == "":
            return "blank", None
        return "valid", value
    if isinstance(value, list):
        if not value:
            return "blank", None
        if not all(isinstance(item, str) for item in value):
            return "invalid", None
        stripped_items = [item.strip() for item in value]
        if all(stripped_items):
            return "valid", value
        if all(item == "" for item in stripped_items):
            return "blank", None
        return "invalid", None
    return "invalid", None


def _iter_label_names(labels: Iterable[str | list[str] | None]) -> Iterable[str]:
    for label in labels:
        if isinstance(label, str):
            yield label
        elif isinstance(label, list):
            yield from label


def _detect_task_type(labels: Sequence[str | list[str]]) -> str:
    if not labels:
        return "not_supervised"
    if any(isinstance(label, list) for label in labels):
        return "multilabel"
    unique = set(label for label in labels if isinstance(label, str))
    if len(unique) == 2:
        return "binary"
    if len(unique) >= 3:
        return "single_label_multiclass"
    return "not_supervised"


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, RATE_DECIMAL_PLACES)


def _class_distribution_items(
    counts: Counter[str],
    total_usable_records: int,
) -> list[ClassDistributionItem]:
    return [
        ClassDistributionItem(
            class_name=label,
            count=count,
            percentage=_rate(count, total_usable_records),
        )
        for label, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _largest_smallest_ratio(items: Sequence[ClassDistributionItem]) -> float | None:
    if len(items) < 2:
        return None
    smallest = items[-1].count
    if smallest == 0:
        return None
    return round(items[0].count / smallest, RATE_DECIMAL_PLACES)


def _imbalance_severity(items: Sequence[ClassDistributionItem]) -> str:
    ratio = _largest_smallest_ratio(items)
    if ratio is None:
        return "not_applicable"
    if ratio < 2:
        return "none"
    if ratio < 5:
        return "mild"
    if ratio < 20:
        return "moderate"
    return "severe"


def _suspicious_label_groups(labels: Iterable[str]) -> list[list[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        normalized = " ".join(label.strip().casefold().split())
        groups[normalized].add(label)
    return [
        sorted(values)
        for _, values in sorted(groups.items())
        if len(values) > 1
    ]


def _stats(values: Sequence[int]) -> TextLengthStatistics:
    if not values:
        return TextLengthStatistics(
            minimum=0,
            mean=0.0,
            median=0.0,
            p90=0,
            p95=0,
            p99=0,
            maximum=0,
        )
    ordered = sorted(values)
    return TextLengthStatistics(
        minimum=ordered[0],
        mean=round(sum(ordered) / len(ordered), RATE_DECIMAL_PLACES),
        median=round(_median(ordered), RATE_DECIMAL_PLACES),
        p90=_percentile(ordered, 90),
        p95=_percentile(ordered, 95),
        p99=_percentile(ordered, 99),
        maximum=ordered[-1],
    )


def _median(ordered: Sequence[int]) -> float:
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _percentile(ordered: Sequence[int], percentile: int) -> int:
    position = math.ceil((percentile / 100) * len(ordered)) - 1
    return ordered[max(0, min(position, len(ordered) - 1))]


def _word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _placeholder_only(text: str, tokens: frozenset[str]) -> bool:
    words = _WORD_PATTERN.findall(text)
    return bool(words) and all(word in tokens for word in words)


def _summarize_text_lengths(
    texts: Sequence[str],
    settings: ClassificationAuditConfig,
) -> TextLengthSummary:
    word_counts = [_word_count(text) for text in texts]
    return TextLengthSummary(
        character_length=_stats([len(text) for text in texts]),
        word_length=_stats(word_counts),
        empty_input_records=sum(1 for text in texts if not text.strip()),
        short_input_records=sum(
            1 for count in word_counts if count < settings.short_text_min_words
        ),
        placeholder_only_records=sum(
            1 for text in texts if _placeholder_only(text, settings.placeholder_only_tokens)
        ),
        short_text_min_words=settings.short_text_min_words,
    )


def _duplicate_input_summary(
    texts: Sequence[str],
    labels: Sequence[str | list[str] | None],
    record_ids: Sequence[str],
) -> DuplicateInputSummary:
    groups: dict[str, list[tuple[str, str | list[str] | None]]] = defaultdict(list)
    for record_id, text, label in zip(record_ids, texts, labels):
        if label is None:
            continue
        groups[text].append((record_id, label))

    duplicated = [group for group in groups.values() if len(group) > 1]
    consistent = 0
    conflicts: list[dict[str, object]] = []
    for text_value, group in groups.items():
        if len(group) < 2:
            continue
        label_keys = {_label_key(label) for _, label in group}
        if len(label_keys) == 1:
            consistent += 1
            continue
        conflicts.append(
            {
                "record_ids": [record_id for record_id, _ in group[:5]],
                "labels": sorted(label_keys),
                "text_fingerprint": _text_fingerprint(text_value),
            }
        )

    return DuplicateInputSummary(
        duplicated_input_groups=len(duplicated),
        groups_with_consistent_labels=consistent,
        groups_with_conflicting_labels=len(conflicts),
        conflicting_examples=conflicts[:10],
    )


def _label_key(label: str | list[str] | None) -> str:
    if isinstance(label, list):
        return "|".join(sorted(label))
    return "" if label is None else label


def _text_fingerprint(text: str) -> str:
    from ticket_pipeline.versioning import sha256_hex

    return sha256_hex(text.encode("utf-8"))


def _input_field_policy(settings: ClassificationAuditConfig) -> list[InputFieldPolicyItem]:
    policies: list[InputFieldPolicyItem] = []
    for field_name in sorted(Ticket.model_fields):
        if field_name in settings.input_fields:
            usage = FieldUsage.ALLOWED_INPUT
            reason = "Customer ticket text available at ticket creation time."
        elif field_name == settings.target_field:
            usage = FieldUsage.TARGET
            reason = "Configured supervised classification label."
        elif field_name in {"queue", "priority", "tags"}:
            usage = FieldUsage.LEAKAGE_RISK
            reason = (
                "Operational label or routing metadata that may encode the "
                "classification target and should not be used as model input."
            )
        elif field_name in {"language_detection", "leakage_check"}:
            usage = FieldUsage.NOT_AVAILABLE_AT_INFERENCE
            reason = "Generated by Phase 1 processing, not original customer text."
        elif field_name == "source_version":
            usage = FieldUsage.IDENTIFIER
            reason = "Dataset/source marker for traceability, not model input."
        else:
            usage = FieldUsage.METADATA_ONLY
            reason = "Metadata retained for analysis but excluded from model input."
        policies.append(
            InputFieldPolicyItem(
                field_name=field_name,
                usage=usage,
                reason=reason,
            )
        )
    return policies


def _classification_leakage_risks(
    policies: Sequence[InputFieldPolicyItem],
) -> list[LeakageRiskItem]:
    return [
        LeakageRiskItem(field_name=item.field_name, reason=item.reason)
        for item in policies
        if item.usage is FieldUsage.LEAKAGE_RISK
    ]


def _warnings(
    *,
    target_summary: TargetSummary,
    distribution: Sequence[ClassDistributionItem],
    rare_classes: Sequence[ClassDistributionItem],
    text_length_summary: TextLengthSummary,
    conflicting_labels: DuplicateInputSummary,
) -> list[str]:
    # Leakage-risk fields (see `_input_field_policy`) are a structural policy
    # about the Ticket schema itself, not a property of this particular
    # dataset, so their presence is reported via `leakage_risks` but does not
    # generate a warning here or affect readiness.
    warnings: list[str] = []
    if target_summary.records_with_missing_labels:
        warnings.append("Some records are missing the configured target field.")
    if target_summary.records_with_blank_labels:
        warnings.append("Some records have blank target labels.")
    if target_summary.records_with_invalid_labels:
        warnings.append("Some records have invalid target labels.")
    if target_summary.suspicious_label_groups:
        warnings.append("Some labels differ only by whitespace or case.")
    if rare_classes:
        warnings.append("Some classes are below the rare-class sample threshold.")
    severity = _imbalance_severity(distribution)
    if severity in {"moderate", "severe"}:
        warnings.append(f"Class imbalance severity is {severity}.")
    if text_length_summary.empty_input_records:
        warnings.append("Some constructed classification inputs are empty.")
    if text_length_summary.short_input_records:
        warnings.append("Some constructed classification inputs are very short.")
    if text_length_summary.placeholder_only_records:
        warnings.append("Some constructed classification inputs contain only placeholders.")
    if conflicting_labels.groups_with_conflicting_labels:
        warnings.append("Identical classification inputs appear with conflicting labels.")
    return sorted(warnings)


def _readiness_status(
    *,
    task_type: str,
    target_summary: TargetSummary,
    warnings: Sequence[str],
) -> DatasetReadinessStatus:
    if (
        task_type == "not_supervised"
        or target_summary.records_with_usable_labels == 0
        or target_summary.unique_class_count < 2
    ):
        return DatasetReadinessStatus.NOT_READY
    if warnings:
        return DatasetReadinessStatus.READY_WITH_WARNINGS
    return DatasetReadinessStatus.READY
