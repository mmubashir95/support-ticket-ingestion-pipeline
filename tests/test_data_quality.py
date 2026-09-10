import json
from datetime import datetime, timezone

import numpy as np

from ticket_pipeline.data_quality import (
    data_quality_report_to_json,
    generate_data_quality_report,
    load_data_quality_report,
    summarize_dataset,
    summarize_deduplication,
    summarize_language,
    summarize_leakage,
    summarize_pii,
    summarize_validation,
    write_data_quality_report,
)
from ticket_pipeline.deduplication import detect_exact_duplicates
from ticket_pipeline.language import (
    LanguageDetector,
    LanguagePrediction,
    annotate_ticket_language,
)
from ticket_pipeline.leakage import (
    LeakageChecker,
    LeakageConfig,
    annotate_ticket_leakage,
)
from ticket_pipeline.loaders import map_source_record
from ticket_pipeline.models import (
    DatasetSourceManifest,
    LanguageDetectionMetadata,
    LanguageDetectionStatus,
    LeakageCheckMetadata,
    LeakageCheckStatus,
    SourceFileManifest,
    Ticket,
)
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import mask_pii_with_metadata
from ticket_pipeline.semantic_deduplication import (
    detect_semantic_duplicate_candidates,
)
from ticket_pipeline.validators import (
    EMPTY_MESSAGE,
    SCHEMA_VALIDATION_FAILED,
    validate_ticket,
    validate_ticket_message,
)
from ticket_pipeline.versioning import (
    capture_processing_config,
    create_dataset_manifest,
    fingerprint_records,
)


class IntegrationEmbeddingModel:
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


def manifest(records: list[object] | None = None, input_count: int = 2):
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
        processed_records=records if records is not None else [],
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )


def test_dataset_counts_rates_and_empty_dataset() -> None:
    summary = summarize_dataset(total_input_records=4, accepted_records=3)

    assert summary.total_input_records == 4
    assert summary.accepted_records == 3
    assert summary.rejected_records == 1
    assert summary.acceptance_rate == 0.75
    assert summary.rejection_rate == 0.25

    empty = summarize_dataset(total_input_records=0, accepted_records=0)
    assert empty.acceptance_rate == 0.0
    assert empty.rejection_rate == 0.0


def test_dataset_rates_cover_all_accepted_and_all_rejected() -> None:
    all_accepted = summarize_dataset(total_input_records=2, accepted_records=2)
    all_rejected = summarize_dataset(total_input_records=2, accepted_records=0)

    assert all_accepted.acceptance_rate == 1.0
    assert all_accepted.rejection_rate == 0.0
    assert all_rejected.acceptance_rate == 0.0
    assert all_rejected.rejection_rate == 1.0


def test_validation_summary_groups_reasons_and_multiple_schema_errors() -> None:
    failures = [
        {
            "reason": SCHEMA_VALIDATION_FAILED,
            "errors": [
                {"loc": ("message",), "type": "missing"},
                {"loc": ("priority",), "type": "literal_error"},
            ],
        },
        {"reason": EMPTY_MESSAGE},
    ]

    summary = summarize_validation(failures)

    assert summary.records_with_validation_issues == 2
    assert summary.validation_issue_events == 3
    assert summary.issue_counts_by_reason == {
        EMPTY_MESSAGE: 1,
        SCHEMA_VALIDATION_FAILED: 1,
    }
    assert summary.schema_error_counts_by_type == {
        "literal_error": 1,
        "missing": 1,
    }
    assert summary.schema_error_counts_by_field == {
        "message": 1,
        "priority": 1,
    }


def test_pii_summary_counts_records_entities_and_types() -> None:
    first_subject = mask_pii_with_metadata("Account ACC-123456")
    first_message = mask_pii_with_metadata(
        "Call +1 555 123 4567 from 192.168.1.20"
    )
    second_message = mask_pii_with_metadata("No sensitive values")

    summary = summarize_pii([[first_subject, first_message], [second_message]])

    assert summary.records_with_pii == 1
    assert summary.total_entities_masked == 3
    assert summary.entities_by_type == {
        "account_id": 1,
        "ip_address": 1,
        "phone": 1,
    }
    assert summary.limitations == []


def test_pii_summary_handles_no_pii_and_missing_metadata_limitation() -> None:
    no_pii = summarize_pii([mask_pii_with_metadata("No sensitive values")])
    no_metadata = summarize_pii(None)

    assert no_pii.records_with_pii == 0
    assert no_pii.total_entities_masked == 0
    assert no_pii.entities_by_type == {}
    assert no_pii.limitations == []
    assert no_metadata.limitations


def test_deduplication_summary_counts_exact_and_semantic_duplicates() -> None:
    exact_results = detect_exact_duplicates(
        [("A", "same"), ("A", "same"), ("B", "near")]
    )
    semantic_results = [
        {
            "record_index": 0,
            "is_semantic_duplicate_candidate": False,
            "candidate_duplicate_of": None,
            "nearest_earlier_record": None,
            "semantic_similarity": None,
            "threshold": 0.85,
            "skipped_exact_duplicate": False,
        },
        {
            "record_index": 1,
            "is_semantic_duplicate_candidate": False,
            "candidate_duplicate_of": None,
            "nearest_earlier_record": None,
            "semantic_similarity": None,
            "threshold": 0.85,
            "skipped_exact_duplicate": True,
        },
        {
            "record_index": 2,
            "is_semantic_duplicate_candidate": True,
            "candidate_duplicate_of": 0,
            "nearest_earlier_record": 0,
            "semantic_similarity": 0.9,
            "threshold": 0.85,
            "skipped_exact_duplicate": False,
        },
    ]

    summary = summarize_deduplication(
        exact_duplicate_results=exact_results,
        semantic_duplicate_results=semantic_results,
    )

    assert summary.exact.duplicate_records_detected == 1
    assert summary.exact.duplicate_groups == 1
    assert summary.semantic.duplicate_candidate_records_detected == 1
    assert summary.semantic.duplicate_candidate_groups == 1
    assert summary.semantic.skipped_exact_duplicate_records == 1
    assert summary.semantic.threshold == 0.85


def test_deduplication_summary_handles_zero_duplicates() -> None:
    summary = summarize_deduplication(
        exact_duplicate_results=detect_exact_duplicates([("A", "one")]),
        semantic_duplicate_results=[],
    )

    assert summary.exact.duplicate_records_detected == 0
    assert summary.exact.duplicate_groups == 0
    assert summary.semantic.duplicate_candidate_records_detected == 0


def test_language_summary_groups_detected_unknown_and_failed() -> None:
    tickets = [
        ticket(
            language_detection=LanguageDetectionMetadata(
                language="en",
                confidence=0.98,
                status=LanguageDetectionStatus.DETECTED,
            )
        ),
        ticket(
            language_detection=LanguageDetectionMetadata(
                language=None,
                confidence=0.55,
                status=LanguageDetectionStatus.UNCERTAIN,
            )
        ),
        ticket(
            language_detection=LanguageDetectionMetadata(
                language=None,
                confidence=None,
                status=LanguageDetectionStatus.FAILED,
            )
        ),
    ]

    summary = summarize_language(tickets)

    assert summary.checks_performed == 3
    assert summary.by_language == {"en": 1, "unknown": 2}
    assert summary.percentages_by_language == {
        "en": 0.333333,
        "unknown": 0.666667,
    }
    assert summary.low_confidence_records == 1
    assert summary.failed_records == 1


def test_leakage_summary_counts_warnings_violations_and_clean_records() -> None:
    tickets = [
        ticket(
            leakage_check=LeakageCheckMetadata(
                status=LeakageCheckStatus.CLEAN
            )
        ),
        ticket(
            leakage_check=LeakageCheckMetadata(
                status=LeakageCheckStatus.WARNING,
                issues=[{"type": "future_field", "field": "answer"}],
            )
        ),
    ]

    summary = summarize_leakage(tickets)

    assert summary.checks_performed == 2
    assert summary.clean_records == 1
    assert summary.warning_records == 1
    assert summary.warnings_detected == 1
    assert summary.violations_detected == 1
    assert summary.issues_by_type == {"future_field": 1}
    assert summary.issues_by_field == {"answer": 1}


def test_clean_leakage_summary_is_valid() -> None:
    summary = summarize_leakage(
        [ticket(leakage_check=LeakageCheckMetadata(status="clean"))]
    )

    assert summary.clean_records == 1
    assert summary.warning_records == 0
    assert summary.violations_detected == 0


def test_report_includes_existing_dataset_version_and_serializes_deterministically(
    tmp_path,
) -> None:
    accepted = [ticket()]
    dataset_manifest = manifest(
        records=[item.model_dump(mode="python") for item in accepted],
        input_count=1,
    )

    first = generate_data_quality_report(
        total_input_records=1,
        accepted_records=accepted,
        dataset_manifest=dataset_manifest,
        preprocessing_metrics={"urls_processed": 2, "records_normalized": 1},
        pii_results=[mask_pii_with_metadata("No sensitive values")],
    )
    second = generate_data_quality_report(
        total_input_records=1,
        accepted_records=accepted,
        dataset_manifest=dataset_manifest,
        preprocessing_metrics={"records_normalized": 1, "urls_processed": 2},
        pii_results=[mask_pii_with_metadata("No sensitive values")],
    )

    assert first == second
    assert first.dataset_version.dataset_version == dataset_manifest.dataset_version
    assert first.dataset_version.input_fingerprint == (
        dataset_manifest.fingerprints.input
    )
    assert "created_at" not in first.dataset_version.model_dump()
    assert data_quality_report_to_json(first) == data_quality_report_to_json(second)

    report_path = tmp_path / "reports" / "data_quality_report.json"
    write_data_quality_report(first, report_path)
    parsed = json.loads(report_path.read_text(encoding="utf-8"))
    assert parsed["dataset_version"]["dataset_version"] == (
        dataset_manifest.dataset_version
    )
    assert load_data_quality_report(report_path) == first


def test_integration_report_aggregates_existing_stage_outputs() -> None:
    source_records = [
        {
            "subject": "Login issue",
            "body": "Cannot log in. Call +1 555 123 4567",
            "answer": "Agent reset the account.",
            "type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "login",
        },
        {
            "subject": "Login issue",
            "body": "Cannot log in. Call +1 555 123 4567",
            "answer": "Agent reset the account.",
            "type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "login",
        },
        {
            "subject": "   ",
            "body": "   ",
            "answer": "",
            "type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "login",
        },
        {
            "subject": "Bad",
            "body": "Missing priority",
            "answer": "",
            "type": "Incident",
            "queue": "Technical Support",
            "language": "en",
            "version": "52",
            "tag_1": "login",
        },
    ]

    validation_failures: list[dict[str, object]] = []
    validated: list[Ticket] = []
    for source_record in source_records:
        ticket_result, schema_failure = validate_ticket(
            map_source_record(source_record)
        )
        if schema_failure is not None:
            validation_failures.append(schema_failure)
            continue
        assert ticket_result is not None
        ticket_result, message_failure = validate_ticket_message(ticket_result)
        if message_failure is not None:
            validation_failures.append(message_failure)
            continue
        assert ticket_result is not None
        validated.append(ticket_result)

    pii_results = []
    processed = []
    for item in validated:
        subject_result = (
            mask_pii_with_metadata(normalize_text(item.subject))
            if item.subject is not None
            else None
        )
        message_result = mask_pii_with_metadata(normalize_text(item.message))
        pii_results.append(
            [result for result in (subject_result, message_result) if result]
        )
        processed.append(
            item.model_copy(
                update={
                    "subject": (
                        subject_result["masked_text"]
                        if subject_result is not None
                        else None
                    ),
                    "message": message_result["masked_text"],
                }
            )
        )

    processed_text = [(item.subject, item.message) for item in processed]
    exact_results = detect_exact_duplicates(processed_text)
    semantic_results = detect_semantic_duplicate_candidates(
        processed_text,
        exact_duplicate_of=[result["duplicate_of"] for result in exact_results],
        model=IntegrationEmbeddingModel(),
    )

    detector = LanguageDetector(backend=EnglishBackend())
    language_annotated = [
        annotate_ticket_language(item, detector) for item in processed
    ]
    checker = LeakageChecker(LeakageConfig())
    accepted = [
        annotate_ticket_leakage(
            item,
            checker,
            source_fields=source_record,
            exact_duplicate=exact_result,
            semantic_duplicate=semantic_result,
        )
        for item, source_record, exact_result, semantic_result in zip(
            language_annotated,
            source_records,
            exact_results,
            semantic_results,
        )
    ]
    dataset_manifest = manifest(
        records=[item.model_dump(mode="python") for item in accepted],
        input_count=len(source_records),
    )

    report = generate_data_quality_report(
        total_input_records=len(source_records),
        accepted_records=accepted,
        dataset_manifest=dataset_manifest,
        validation_failures=validation_failures,
        exact_duplicate_results=exact_results,
        semantic_duplicate_results=semantic_results,
        pii_results=pii_results,
    )

    assert report.dataset_summary.total_input_records == 4
    assert report.dataset_summary.accepted_records == 2
    assert report.dataset_summary.rejected_records == 2
    assert report.validation_summary.issue_counts_by_reason == {
        EMPTY_MESSAGE: 1,
        SCHEMA_VALIDATION_FAILED: 1,
    }
    assert report.pii_summary.records_with_pii == 2
    assert report.pii_summary.entities_by_type == {"phone": 2}
    assert report.deduplication_summary.exact.duplicate_records_detected == 1
    assert report.language_summary.by_language == {"en": 2}
    assert report.leakage_summary.warning_records == 2
    assert report.leakage_summary.issues_by_field == {"answer": 2}
    assert report.dataset_version.output_fingerprint == fingerprint_records(
        accepted
    )
