from datetime import datetime, timezone

import numpy as np

from ticket_pipeline.deduplication import detect_exact_duplicates
from ticket_pipeline.language import (
    LanguageDetectionConfig,
    LanguageDetector,
    LanguagePrediction,
    annotate_ticket_language,
)
from ticket_pipeline.leakage import (
    LeakageChecker,
    LeakageConfig,
    annotate_ticket_leakage,
)
from ticket_pipeline.loaders import load_csv
from ticket_pipeline.models import Ticket
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import mask_pii
from ticket_pipeline.semantic_deduplication import (
    detect_semantic_duplicate_candidates,
)
from ticket_pipeline.validators import validate_ticket, validate_ticket_message
from ticket_pipeline.versioning import (
    capture_processing_config,
    create_dataset_manifest,
    fingerprint_source_files,
    load_dataset_manifest,
    write_dataset_manifest,
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
        assert len(sentences) == 2
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class EnglishBackend:
    def predict(self, text: str) -> LanguagePrediction:
        return LanguagePrediction("en", 0.98)


def test_current_pipeline_build_produces_reproducible_dataset_manifest(
    tmp_path,
) -> None:
    source_path = "tests/fixtures/tickets_sample.csv"
    canonical_records = load_csv(source_path)

    validated: list[Ticket] = []
    for record in canonical_records:
        ticket, schema_failure = validate_ticket(record)
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
    processed_text = [(ticket.subject, ticket.message) for ticket in processed]
    exact_results = detect_exact_duplicates(processed_text)
    semantic_results = detect_semantic_duplicate_candidates(
        processed_text,
        exact_duplicate_of=[result["duplicate_of"] for result in exact_results],
        model=IntegrationEmbeddingModel(),
    )

    language_config = LanguageDetectionConfig()
    detector = LanguageDetector(language_config, backend=EnglishBackend())
    language_annotated = [
        annotate_ticket_language(ticket, detector) for ticket in processed
    ]
    leakage_config = LeakageConfig()
    checker = LeakageChecker(leakage_config)
    versionable_records = [
        annotate_ticket_leakage(
            ticket,
            checker,
            source_fields={"answer": "Agent response present"},
            exact_duplicate=exact_result,
            semantic_duplicate=semantic_result,
        )
        for ticket, exact_result, semantic_result in zip(
            language_annotated,
            exact_results,
            semantic_results,
        )
    ]
    before_versioning = [
        ticket.model_dump(mode="python") for ticket in versionable_records
    ]

    source = fingerprint_source_files([source_path])
    processing_config = capture_processing_config(
        semantic_deduplication_enabled=True,
        semantic_model_name="test-integration-embedding-v1",
        language_config=language_config,
        leakage_config=leakage_config,
    )
    first = create_dataset_manifest(
        source=source,
        processing_config=processing_config,
        input_record_count=len(canonical_records),
        processed_records=versionable_records,
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    second = create_dataset_manifest(
        source=source,
        processing_config=processing_config,
        input_record_count=len(canonical_records),
        processed_records=versionable_records,
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )

    assert first.dataset_version == second.dataset_version
    assert first.fingerprints == second.fingerprints
    assert first.source.files[0].name == "tickets_sample.csv"
    assert first.counts.input == 2
    assert first.counts.processed == 2
    assert first.pipeline_version == "0.1.0"
    assert first.processing_config.semantic_deduplication is not None
    assert (
        first.processing_config.semantic_deduplication.model_name
        == "test-integration-embedding-v1"
    )
    assert [
        ticket.model_dump(mode="python") for ticket in versionable_records
    ] == before_versioning

    reordered = create_dataset_manifest(
        source=source,
        processing_config=processing_config,
        input_record_count=len(canonical_records),
        processed_records=list(reversed(versionable_records)),
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    assert reordered.fingerprints.output != first.fingerprints.output
    assert reordered.dataset_version != first.dataset_version

    manifest_path = tmp_path / "dataset_manifest.json"
    write_dataset_manifest(first, manifest_path)
    assert load_dataset_manifest(manifest_path) == first
