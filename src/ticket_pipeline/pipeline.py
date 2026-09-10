"""Phase 1 orchestration for the support-ticket ingestion pipeline."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from ticket_pipeline.data_quality import generate_data_quality_report
from ticket_pipeline.deduplication import (
    ExactDuplicateResult,
    detect_exact_duplicates,
)
from ticket_pipeline.language import (
    LanguageDetectionConfig,
    LanguageDetector,
    LanguageIdentifier,
    annotate_ticket_language,
)
from ticket_pipeline.leakage import (
    LeakageChecker,
    LeakageConfig,
    annotate_ticket_leakage,
)
from ticket_pipeline.loaders import load_source_records, map_source_record
from ticket_pipeline.models import DataQualityReport, DatasetManifest, Ticket
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import PiiMaskingResult, mask_pii_with_metadata
from ticket_pipeline.semantic_deduplication import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    DEFAULT_TOP_K,
    EmbeddingModel,
    SemanticDuplicateResult,
    detect_semantic_duplicate_candidates,
)
from ticket_pipeline.validators import (
    MessageValidationFailure,
    SchemaValidationFailure,
    validate_ticket,
    validate_ticket_message,
)
from ticket_pipeline.versioning import (
    capture_processing_config,
    create_dataset_manifest,
    fingerprint_source_files,
)


ValidationFailure = SchemaValidationFailure | MessageValidationFailure
T = TypeVar("T")


class PipelineExecutionError(RuntimeError):
    """Fatal orchestration failure that is not a record-level rejection."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration composed from existing stage-level settings."""

    semantic_deduplication_enabled: bool = True
    semantic_model_name: str = DEFAULT_EMBEDDING_MODEL
    semantic_threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD
    semantic_top_k: int = DEFAULT_TOP_K
    semantic_batch_size: int = DEFAULT_BATCH_SIZE
    semantic_model: EmbeddingModel | None = None
    language_config: LanguageDetectionConfig | None = None
    language_backend: LanguageIdentifier | None = None
    leakage_config: LeakageConfig | None = None
    pipeline_version: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    """One source record rejected by existing validation stages."""

    record_index: int
    canonical_record: Mapping[str, object]
    failure: ValidationFailure


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Complete in-memory result for a Phase 1 pipeline run."""

    accepted_records: list[Ticket]
    rejected_records: list[RejectedRecord]
    dataset_manifest: DatasetManifest
    data_quality_report: DataQualityReport


@dataclass(frozen=True, slots=True)
class _ActiveRecord:
    original_index: int
    source_record: Mapping[str, object]
    ticket: Ticket


def run_pipeline(
    source: str | Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the complete Phase 1 pipeline over one CSV or JSON source file."""

    settings = config or PipelineConfig()
    source_path = Path(source)

    source_records = _run_stage(
        "loading source records",
        lambda: load_source_records(source_path),
    )
    active_records, rejected_records = _run_stage(
        "validation",
        lambda: _validate_records(source_records),
    )
    processed_records, pii_results = _run_stage(
        "preprocessing",
        lambda: _preprocess_records(active_records),
    )
    processed_text = [
        (record.ticket.subject, record.ticket.message)
        for record in processed_records
    ]
    exact_results = _run_stage(
        "exact deduplication",
        lambda: detect_exact_duplicates(processed_text),
    )
    semantic_results = _run_semantic_deduplication(
        processed_text,
        exact_results,
        settings,
    )
    language_records = _run_stage(
        "language detection",
        lambda: _annotate_languages(processed_records, settings),
    )
    accepted_records = _run_stage(
        "leakage checks",
        lambda: _annotate_leakage(
            language_records,
            exact_results,
            semantic_results,
            settings,
        ),
    )
    manifest = _create_manifest(
        source_path,
        len(source_records),
        accepted_records,
        settings,
    )
    report = _run_stage(
        "data-quality reporting",
        lambda: generate_data_quality_report(
            total_input_records=len(source_records),
            accepted_records=accepted_records,
            dataset_manifest=manifest,
            validation_failures=[
                rejected.failure for rejected in rejected_records
            ],
            exact_duplicate_results=exact_results,
            semantic_duplicate_results=semantic_results,
            pii_results=pii_results,
        ),
    )

    return PipelineResult(
        accepted_records=accepted_records,
        rejected_records=rejected_records,
        dataset_manifest=manifest,
        data_quality_report=report,
    )


def _run_stage(stage_name: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except PipelineExecutionError:
        raise
    except Exception as error:
        raise PipelineExecutionError(
            f"Pipeline failed during {stage_name}."
        ) from error


def _validate_records(
    source_records: Sequence[Mapping[str, object]],
) -> tuple[list[_ActiveRecord], list[RejectedRecord]]:
    active_records: list[_ActiveRecord] = []
    rejected_records: list[RejectedRecord] = []

    for index, source_record in enumerate(source_records):
        canonical_record = map_source_record(dict(source_record))
        ticket, schema_failure = validate_ticket(canonical_record)
        if schema_failure is not None:
            rejected_records.append(
                RejectedRecord(
                    record_index=index,
                    canonical_record=canonical_record,
                    failure=schema_failure,
                )
            )
            continue

        if ticket is None:
            raise PipelineExecutionError(
                "Ticket validation returned neither ticket nor failure."
            )

        ticket, message_failure = validate_ticket_message(ticket)
        if message_failure is not None:
            rejected_records.append(
                RejectedRecord(
                    record_index=index,
                    canonical_record=canonical_record,
                    failure=message_failure,
                )
            )
            continue

        if ticket is None:
            raise PipelineExecutionError(
                "Message validation returned neither ticket nor failure."
            )

        active_records.append(
            _ActiveRecord(
                original_index=index,
                source_record=source_record,
                ticket=ticket,
            )
        )

    return active_records, rejected_records


def _preprocess_records(
    records: Sequence[_ActiveRecord],
) -> tuple[list[_ActiveRecord], list[list[PiiMaskingResult]]]:
    processed_records: list[_ActiveRecord] = []
    pii_results: list[list[PiiMaskingResult]] = []

    for record in records:
        record_pii_results: list[PiiMaskingResult] = []
        subject_result = None
        if record.ticket.subject is not None:
            subject_result = mask_pii_with_metadata(
                normalize_text(record.ticket.subject)
            )
            record_pii_results.append(subject_result)
        message_result = mask_pii_with_metadata(
            normalize_text(record.ticket.message)
        )
        record_pii_results.append(message_result)
        pii_results.append(record_pii_results)
        processed_records.append(
            _ActiveRecord(
                original_index=record.original_index,
                source_record=record.source_record,
                ticket=record.ticket.model_copy(
                    update={
                        "subject": (
                            subject_result["masked_text"]
                            if subject_result is not None
                            else None
                        ),
                        "message": message_result["masked_text"],
                    }
                ),
            )
        )

    return processed_records, pii_results


def _run_semantic_deduplication(
    processed_text: Sequence[tuple[str | None, str]],
    exact_results: Sequence[ExactDuplicateResult],
    settings: PipelineConfig,
) -> list[SemanticDuplicateResult]:
    if not settings.semantic_deduplication_enabled:
        return []

    return _run_stage(
        "semantic deduplication",
        lambda: detect_semantic_duplicate_candidates(
            processed_text,
            exact_duplicate_of=[
                result["duplicate_of"] for result in exact_results
            ],
            threshold=settings.semantic_threshold,
            top_k=settings.semantic_top_k,
            batch_size=settings.semantic_batch_size,
            model_name=settings.semantic_model_name,
            model=settings.semantic_model,
        ),
    )


def _annotate_languages(
    records: Sequence[_ActiveRecord],
    settings: PipelineConfig,
) -> list[_ActiveRecord]:
    if not records:
        return []

    detector = _run_stage(
        "language detector initialization",
        lambda: LanguageDetector(
            settings.language_config,
            backend=settings.language_backend,
        ),
    )
    return [
        _ActiveRecord(
            original_index=record.original_index,
            source_record=record.source_record,
            ticket=annotate_ticket_language(record.ticket, detector),
        )
        for record in records
    ]


def _annotate_leakage(
    records: Sequence[_ActiveRecord],
    exact_results: Sequence[ExactDuplicateResult],
    semantic_results: Sequence[SemanticDuplicateResult],
    settings: PipelineConfig,
) -> list[Ticket]:
    if not records:
        return []

    checker = _run_stage(
        "leakage checker initialization",
        lambda: LeakageChecker(settings.leakage_config),
    )
    semantic_by_record_index = {
        result["record_index"]: result for result in semantic_results
    }

    accepted_records: list[Ticket] = []
    for processed_index, record in enumerate(records):
        semantic_result = semantic_by_record_index.get(processed_index)
        accepted_records.append(
            annotate_ticket_leakage(
                record.ticket,
                checker,
                source_fields=record.source_record,
                exact_duplicate=exact_results[processed_index],
                semantic_duplicate=semantic_result,
            )
        )
    return accepted_records


def _create_manifest(
    source_path: Path,
    total_input_records: int,
    accepted_records: Sequence[Ticket],
    settings: PipelineConfig,
) -> DatasetManifest:
    return _run_stage(
        "dataset versioning",
        lambda: create_dataset_manifest(
            source=fingerprint_source_files([source_path]),
            processing_config=capture_processing_config(
                semantic_deduplication_enabled=(
                    settings.semantic_deduplication_enabled
                ),
                semantic_model_name=settings.semantic_model_name,
                semantic_threshold=settings.semantic_threshold,
                semantic_top_k=settings.semantic_top_k,
                semantic_batch_size=settings.semantic_batch_size,
                language_config=settings.language_config,
                leakage_config=settings.leakage_config,
            ),
            input_record_count=total_input_records,
            processed_records=accepted_records,
            pipeline_version=settings.pipeline_version,
            created_at=settings.created_at,
        ),
    )
