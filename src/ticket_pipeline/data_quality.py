"""Deterministic aggregation of existing pipeline data-quality metadata."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ticket_pipeline.deduplication import ExactDuplicateResult
from ticket_pipeline.models import (
    DataQualityReport,
    DatasetManifest,
    DatasetSummary,
    DatasetVersionSummary,
    DeduplicationSummary,
    ExactDeduplicationSummary,
    LanguageDetectionStatus,
    LanguageSummary,
    LeakageCheckStatus,
    LeakageSummary,
    PiiSummary,
    PreprocessingSummary,
    SemanticDeduplicationSummary,
    Ticket,
    ValidationSummary,
)
from ticket_pipeline.pii import PiiMaskingResult
from ticket_pipeline.semantic_deduplication import SemanticDuplicateResult


RATE_DECIMAL_PLACES = 6
_NO_PREPROCESSING_METADATA_LIMITATION = (
    "Preprocessing stages currently expose text output but no transformation "
    "event counters; only caller-supplied metrics can be reported."
)
_NO_PII_METADATA_LIMITATION = (
    "PII masking counts require metadata captured during masking; processed "
    "ticket text alone is not inspected to infer PII counts."
)


def _rate(part: int, total: int) -> float:
    """Return a deterministic fraction rounded to six decimal places."""

    if total == 0:
        return 0.0
    return round(part / total, RATE_DECIMAL_PLACES)


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def _sorted_rates(rates: Mapping[str, float]) -> dict[str, float]:
    return {key: rates[key] for key in sorted(rates)}


def _validation_failure_reason(failure: Mapping[str, object]) -> str:
    reason = failure.get("reason")
    if not isinstance(reason, str) or not reason:
        return "UNKNOWN_VALIDATION_FAILURE"
    return reason


def _schema_error_field(error: Mapping[str, object]) -> str:
    location = error.get("loc")
    if isinstance(location, (list, tuple)):
        return ".".join(str(part) for part in location)
    if isinstance(location, str):
        return location
    return "unknown"


def summarize_dataset(
    *,
    total_input_records: int,
    accepted_records: int,
) -> DatasetSummary:
    """Summarize record acceptance using total-input as the denominator."""

    rejected_records = total_input_records - accepted_records
    return DatasetSummary(
        total_input_records=total_input_records,
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        acceptance_rate=_rate(accepted_records, total_input_records),
        rejection_rate=_rate(rejected_records, total_input_records),
    )


def summarize_validation(
    validation_failures: Sequence[Mapping[str, object]] | None = None,
) -> ValidationSummary:
    """Aggregate existing validation failure reasons and Pydantic errors.

    Reason counts are per rejected record. ``validation_issue_events`` counts
    individual validation events, so a schema failure with multiple Pydantic
    errors contributes more than one event.
    """

    failures = list(validation_failures or [])
    reason_counts: Counter[str] = Counter()
    schema_error_type_counts: Counter[str] = Counter()
    schema_error_field_counts: Counter[str] = Counter()
    validation_issue_events = 0

    for failure in failures:
        reason = _validation_failure_reason(failure)
        reason_counts[reason] += 1
        errors = failure.get("errors")
        if isinstance(errors, list):
            validation_issue_events += len(errors)
            for error in errors:
                if not isinstance(error, Mapping):
                    continue
                error_type = error.get("type")
                if isinstance(error_type, str) and error_type:
                    schema_error_type_counts[error_type] += 1
                schema_error_field_counts[_schema_error_field(error)] += 1
        else:
            validation_issue_events += 1

    return ValidationSummary(
        records_with_validation_issues=len(failures),
        validation_issue_events=validation_issue_events,
        issue_counts_by_reason=_sorted_counts(reason_counts),
        schema_error_counts_by_type=_sorted_counts(schema_error_type_counts),
        schema_error_counts_by_field=_sorted_counts(schema_error_field_counts),
    )


def summarize_preprocessing(
    preprocessing_metrics: Mapping[str, int] | None = None,
) -> PreprocessingSummary:
    """Return caller-supplied preprocessing counters, if any."""

    metrics = dict(preprocessing_metrics or {})
    limitations = [] if metrics else [_NO_PREPROCESSING_METADATA_LIMITATION]
    return PreprocessingSummary(
        metrics=_sorted_counts(metrics),
        limitations=limitations,
    )


PiiRecordMaskingResults = PiiMaskingResult | Sequence[PiiMaskingResult]


def _pii_result_sequence(
    pii_record_results: PiiRecordMaskingResults,
) -> list[PiiMaskingResult]:
    if isinstance(pii_record_results, Mapping):
        return [pii_record_results]
    return list(pii_record_results)


def summarize_pii(
    pii_results: Sequence[PiiRecordMaskingResults] | None = None,
) -> PiiSummary:
    """Aggregate PII masking metadata captured by Step 7.

    Each top-level item represents one ticket record. If a record's subject
    and message were masked separately, pass both field results as a nested
    sequence so ``records_with_pii`` remains a record count, not a field count.
    """

    if pii_results is None:
        return PiiSummary(
            records_with_pii=0,
            total_entities_masked=0,
            entities_by_type={},
            limitations=[_NO_PII_METADATA_LIMITATION],
        )

    entities_by_type: Counter[str] = Counter()
    records_with_pii = 0
    total_entities_masked = 0
    for record_results in pii_results:
        record_entity_count = 0
        for result in _pii_result_sequence(record_results):
            entity_count = int(result["total_entities_masked"])
            record_entity_count += entity_count
            total_entities_masked += entity_count
            entities_by_type.update(result["entities_by_type"])
        if record_entity_count:
            records_with_pii += 1

    return PiiSummary(
        records_with_pii=records_with_pii,
        total_entities_masked=total_entities_masked,
        entities_by_type=_sorted_counts(entities_by_type),
        limitations=[],
    )


def summarize_deduplication(
    *,
    exact_duplicate_results: Sequence[ExactDuplicateResult] | None = None,
    semantic_duplicate_results: Sequence[SemanticDuplicateResult] | None = None,
) -> DeduplicationSummary:
    """Aggregate Step 8 and Step 9 result objects without recomputation."""

    exact_results = list(exact_duplicate_results or [])
    exact_duplicate_links = [
        result["duplicate_of"]
        for result in exact_results
        if result["is_exact_duplicate"]
    ]

    semantic_results = list(semantic_duplicate_results or [])
    semantic_duplicate_links = [
        result["candidate_duplicate_of"]
        for result in semantic_results
        if result["is_semantic_duplicate_candidate"]
    ]
    thresholds = {
        result["threshold"]
        for result in semantic_results
        if result["threshold"] is not None
    }
    threshold = next(iter(thresholds)) if len(thresholds) == 1 else None

    return DeduplicationSummary(
        exact=ExactDeduplicationSummary(
            records_evaluated=len(exact_results),
            duplicate_records_detected=len(exact_duplicate_links),
            duplicate_groups=len(
                {link for link in exact_duplicate_links if link is not None}
            ),
        ),
        semantic=SemanticDeduplicationSummary(
            records_evaluated=len(semantic_results),
            duplicate_candidate_records_detected=len(semantic_duplicate_links),
            duplicate_candidate_groups=len(
                {link for link in semantic_duplicate_links if link is not None}
            ),
            skipped_exact_duplicate_records=sum(
                1
                for result in semantic_results
                if result["skipped_exact_duplicate"]
            ),
            threshold=threshold,
        ),
    )


def summarize_language(tickets: Sequence[Ticket]) -> LanguageSummary:
    """Aggregate Step 10 metadata already attached to accepted tickets."""

    language_counts: Counter[str] = Counter()
    checks_performed = 0
    detected_records = 0
    unknown_records = 0
    low_confidence_records = 0
    failed_records = 0

    for ticket in tickets:
        detection = ticket.language_detection
        if detection is None:
            continue
        checks_performed += 1
        if detection.status is LanguageDetectionStatus.DETECTED:
            detected_records += 1
            if detection.language is not None:
                language_counts[detection.language] += 1
        elif detection.status is LanguageDetectionStatus.UNCERTAIN:
            unknown_records += 1
            low_confidence_records += int(detection.confidence is not None)
            language_counts["unknown"] += 1
        elif detection.status is LanguageDetectionStatus.FAILED:
            failed_records += 1
            language_counts["unknown"] += 1

    percentages = {
        language: _rate(count, checks_performed)
        for language, count in language_counts.items()
    }
    return LanguageSummary(
        checks_performed=checks_performed,
        detected_records=detected_records,
        unknown_records=unknown_records,
        low_confidence_records=low_confidence_records,
        failed_records=failed_records,
        by_language=_sorted_counts(language_counts),
        percentages_by_language=_sorted_rates(percentages),
    )


def summarize_leakage(tickets: Sequence[Ticket]) -> LeakageSummary:
    """Aggregate Step 11 metadata without including leaked raw field values."""

    checks_performed = 0
    clean_records = 0
    warning_records = 0
    failed_records = 0
    issues_by_type: Counter[str] = Counter()
    issues_by_field: Counter[str] = Counter()

    for ticket in tickets:
        check = ticket.leakage_check
        if check is None:
            continue
        checks_performed += 1
        if check.status is LeakageCheckStatus.CLEAN:
            clean_records += 1
        elif check.status is LeakageCheckStatus.WARNING:
            warning_records += 1
        elif check.status is LeakageCheckStatus.FAILED:
            failed_records += 1

        for issue in check.issues:
            issues_by_type[issue.type.value] += 1
            issues_by_field[issue.field] += 1

    violations_detected = sum(issues_by_type.values())
    return LeakageSummary(
        checks_performed=checks_performed,
        clean_records=clean_records,
        warning_records=warning_records,
        failed_records=failed_records,
        violations_detected=violations_detected,
        warnings_detected=warning_records,
        issues_by_type=_sorted_counts(issues_by_type),
        issues_by_field=_sorted_counts(issues_by_field),
    )


def summarize_dataset_version(manifest: DatasetManifest) -> DatasetVersionSummary:
    """Copy the authoritative Step 12 identity into the report."""

    return DatasetVersionSummary(
        dataset_version=manifest.dataset_version,
        pipeline_version=manifest.pipeline_version,
        input_fingerprint=manifest.fingerprints.input,
        config_fingerprint=manifest.fingerprints.config,
        output_fingerprint=manifest.fingerprints.output,
    )


def generate_data_quality_report(
    *,
    total_input_records: int,
    accepted_records: Sequence[Ticket],
    dataset_manifest: DatasetManifest,
    validation_failures: Sequence[Mapping[str, object]] | None = None,
    exact_duplicate_results: Sequence[ExactDuplicateResult] | None = None,
    semantic_duplicate_results: Sequence[SemanticDuplicateResult] | None = None,
    preprocessing_metrics: Mapping[str, int] | None = None,
    pii_results: Sequence[PiiRecordMaskingResults] | None = None,
) -> DataQualityReport:
    """Build a deterministic report from prior pipeline stage outputs."""

    accepted = list(accepted_records)
    return DataQualityReport(
        dataset_summary=summarize_dataset(
            total_input_records=total_input_records,
            accepted_records=len(accepted),
        ),
        validation_summary=summarize_validation(validation_failures),
        preprocessing_summary=summarize_preprocessing(preprocessing_metrics),
        pii_summary=summarize_pii(pii_results),
        deduplication_summary=summarize_deduplication(
            exact_duplicate_results=exact_duplicate_results,
            semantic_duplicate_results=semantic_duplicate_results,
        ),
        language_summary=summarize_language(accepted),
        leakage_summary=summarize_leakage(accepted),
        dataset_version=summarize_dataset_version(dataset_manifest),
    )


def data_quality_report_to_dict(report: DataQualityReport) -> dict[str, Any]:
    """Return the JSON-native representation used for persistence."""

    return report.model_dump(mode="json")


def data_quality_report_to_json(report: DataQualityReport) -> str:
    """Serialize a report as stable, readable JSON."""

    return json.dumps(
        data_quality_report_to_dict(report),
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )


def write_data_quality_report(
    report: DataQualityReport,
    path: str | Path,
) -> None:
    """Write a stable JSON data-quality report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{data_quality_report_to_json(report)}\n",
        encoding="utf-8",
    )


def load_data_quality_report(path: str | Path) -> DataQualityReport:
    """Load and validate a previously written data-quality report."""

    with Path(path).open(encoding="utf-8") as report_file:
        return DataQualityReport.model_validate(json.load(report_file))
