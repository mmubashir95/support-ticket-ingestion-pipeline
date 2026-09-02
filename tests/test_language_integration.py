import numpy as np

from ticket_pipeline.deduplication import detect_exact_duplicates
from ticket_pipeline.language import LanguageDetector, annotate_ticket_language
from ticket_pipeline.loaders import map_source_record
from ticket_pipeline.models import LanguageDetectionStatus, Ticket
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import mask_pii
from ticket_pipeline.semantic_deduplication import (
    build_semantic_text,
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
        assert sentences == [
            "Login problem\nI cannot log into my account. Email <EMAIL>",
            "Problema de acceso\nNo puedo iniciar sesión en mi cuenta.",
        ]
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_ticket_flow_through_language_detection_preserves_cleaned_records() -> None:
    source_records = [
        {
            "subject": "Login problem",
            "body": "I cannot log into my account. Email user@example.com",
            "type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "version": "52",
            "tag_1": "login",
        },
        {
            "subject": "Problema de acceso",
            "body": "No puedo iniciar sesión en mi cuenta.",
            "type": "Incident",
            "queue": "Technical Support",
            "priority": "high",
            "language": "es",
            "version": "52",
            "tag_1": "login",
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
    detector = LanguageDetector()
    annotated = [annotate_ticket_language(ticket, detector) for ticket in processed]

    assert all(not result["is_exact_duplicate"] for result in exact_results)
    assert all(
        not result["is_semantic_duplicate_candidate"]
        for result in semantic_results
    )
    assert [(ticket.subject, ticket.message) for ticket in annotated] == cleaned_text
    assert annotated[0].message.endswith("Email <EMAIL>")
    assert annotated[0].language_detection is not None
    assert annotated[0].language_detection.language == "en"
    assert annotated[1].language_detection is not None
    assert annotated[1].language_detection.language == "es"
    assert annotated[1].language_detection.status is LanguageDetectionStatus.DETECTED
    assert annotated[1].language == "es"
    assert annotated[1].tags == ["login"]
