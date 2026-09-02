import numpy as np

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
from ticket_pipeline.models import LeakageCheckStatus, LeakageIssueType, Ticket
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import mask_pii
from ticket_pipeline.semantic_deduplication import (
    detect_semantic_duplicate_candidates,
)
from ticket_pipeline.validators import validate_ticket, validate_ticket_message


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
        assert len(sentences) == 2
        return np.asarray(
            [[1.0, 0.0], [0.99, 0.1]],
            dtype=np.float32,
        )


class EnglishBackend:
    def predict(self, text: str) -> LanguagePrediction:
        return LanguagePrediction("en", 0.98)


def test_current_pipeline_stages_feed_leakage_metadata_without_recomputation() -> None:
    source_records = [
        {
            "subject": "Refund not received",
            "body": "I requested a refund yesterday but it still has not arrived.",
            "answer": "Refund completed",
            "type": "Incident",
            "queue": "Billing and Payments",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "refund",
            "created_at": "2026-09-01T09:00:00Z",
            "closed_at": "2026-09-03T09:00:00Z",
            "resolution_text": "Refund completed",
            "conversation_id": "CONV-100",
        },
        {
            "subject": "Refund not received",
            "body": "I requested a refund yesterday but it still has not arrived.",
            "answer": "Refund completed",
            "type": "Incident",
            "queue": "Billing and Payments",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "refund",
            "created_at": "2026-09-01T09:05:00Z",
            "closed_at": "2026-09-03T09:00:00Z",
            "resolution_text": "Refund completed",
            "conversation_id": "CONV-100",
        },
        {
            "subject": "Refund is still missing",
            "body": "The refund I asked for yesterday has not reached me yet.",
            "answer": "Billing confirmed the refund",
            "type": "Incident",
            "queue": "Billing and Payments",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "refund",
            "created_at": "2026-09-01T10:00:00Z",
            "closed_at": "2026-09-03T11:00:00Z",
            "resolution_text": "Refund completed",
            "conversation_id": "CONV-101",
        },
    ]

    validated: list[Ticket] = []
    for source_record in source_records:
        ticket, schema_failure = validate_ticket(map_source_record(source_record))
        assert schema_failure is None
        assert ticket is not None
        ticket, message_failure = validate_ticket_message(ticket)
        assert message_failure is None
        assert ticket is not None
        validated.append(ticket)

    processed = [
        ticket.model_copy(
            update={
                "subject": (
                    mask_pii(normalize_text(ticket.subject))
                    if ticket.subject is not None
                    else None
                ),
                "message": mask_pii(normalize_text(ticket.message)),
            }
        )
        for ticket in validated
    ]
    cleaned_text = [(ticket.subject, ticket.message) for ticket in processed]
    exact_results = detect_exact_duplicates(cleaned_text)
    semantic_results = detect_semantic_duplicate_candidates(
        cleaned_text,
        exact_duplicate_of=[result["duplicate_of"] for result in exact_results],
        model=IntegrationEmbeddingModel(),
    )

    language_detector = LanguageDetector(backend=EnglishBackend())
    language_annotated = [
        annotate_ticket_language(ticket, language_detector)
        for ticket in processed
    ]
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields={"answer", "closed_at", "resolution_text"},
            target_fields={"needs_escalation"},
            target_proxy_fields={"escalation_result"},
            group_fields={"conversation_id"},
        )
    )
    leakage_annotated = [
        annotate_ticket_leakage(
            ticket,
            checker,
            source_fields=source_record,
            exact_duplicate=exact_result,
            semantic_duplicate=semantic_result,
        )
        for ticket, source_record, exact_result, semantic_result in zip(
            language_annotated,
            source_records,
            exact_results,
            semantic_results,
        )
    ]

    assert [(ticket.subject, ticket.message) for ticket in leakage_annotated] == (
        cleaned_text
    )
    assert all(ticket.language_detection is not None for ticket in leakage_annotated)
    assert all(ticket.leakage_check is not None for ticket in leakage_annotated)
    assert all(
        ticket.leakage_check.status is LeakageCheckStatus.WARNING
        for ticket in leakage_annotated
        if ticket.leakage_check is not None
    )
    first_leakage = leakage_annotated[0].leakage_check
    exact_leakage = leakage_annotated[1].leakage_check
    semantic_leakage = leakage_annotated[2].leakage_check
    assert first_leakage is not None
    assert exact_leakage is not None
    assert semantic_leakage is not None
    assert {
        issue.type for issue in first_leakage.issues
    } == {LeakageIssueType.FUTURE_FIELD}
    assert {
        issue.field for issue in first_leakage.issues
    } == {"answer", "closed_at", "resolution_text"}
    assert exact_leakage.grouping.exact_duplicate_group_id == 0
    assert semantic_leakage.grouping.semantic_duplicate_group_id == 0
    assert first_leakage.grouping.identifiers == {
        "conversation_id": "CONV-100"
    }
