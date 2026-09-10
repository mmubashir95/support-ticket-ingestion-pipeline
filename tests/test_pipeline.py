import json
from datetime import datetime, timezone

import numpy as np
import pytest

from ticket_pipeline.data_quality import generate_data_quality_report
from ticket_pipeline.language import (
    LanguageDetectionConfig,
    LanguagePrediction,
)
from ticket_pipeline.leakage import LeakageConfig
from ticket_pipeline.loaders import load_records, load_source_records
from ticket_pipeline.pipeline import (
    PipelineConfig,
    PipelineExecutionError,
    PipelineResult,
    RejectedRecord,
    run_pipeline,
)
from ticket_pipeline.validators import EMPTY_MESSAGE, SCHEMA_VALIDATION_FAILED


class DeterministicEmbeddingModel:
    def __init__(self) -> None:
        self.batch_size: int | None = None
        self.encoded_texts: list[str] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        self.batch_size = batch_size
        self.encoded_texts = sentences
        assert normalize_embeddings is True
        assert show_progress_bar is False
        assert convert_to_numpy is True
        vectors = []
        for sentence in sentences:
            if "Payment problem" in sentence:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class FixedLanguageBackend:
    def __init__(self, confidence: float = 0.98) -> None:
        self.confidence = confidence
        self.texts: list[str] = []

    def predict(self, text: str) -> LanguagePrediction:
        self.texts.append(text)
        return LanguagePrediction("en", self.confidence)


class NoisyLanguageBackend:
    """Backend whose confidence jitters in its least-significant bits.

    Real multithreaded detectors return a slightly different float each run
    for the same text. The pipeline must quantize that value before storing
    it so the content-addressed dataset version stays reproducible.
    """

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, text: str) -> LanguagePrediction:
        self.calls += 1
        jitter = (self.calls % 7) * 1e-11
        return LanguagePrediction("en", 0.9312345678 + jitter)


def write_csv(path, rows: list[dict[str, str]]) -> None:
    fields = [
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
        "version",
        "tag_1",
        "resolution_text",
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = [row.get(field, "") for field in fields]
        lines.append(",".join(f'"{value}"' for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def valid_row(**updates: str) -> dict[str, str]:
    row = {
        "subject": "Login issue",
        "body": "Cannot log in to my account.",
        "answer": "",
        "type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "version": "52",
        "tag_1": "login",
        "resolution_text": "",
    }
    row.update(updates)
    return row


def config(
    *,
    semantic_model: DeterministicEmbeddingModel | None = None,
    language_backend: FixedLanguageBackend | None = None,
    semantic_threshold: float = 0.9,
    semantic_batch_size: int = 2,
    semantic_model_name: str = "test-embedding-v1",
    leakage_config: LeakageConfig | None = None,
) -> PipelineConfig:
    return PipelineConfig(
        semantic_model=semantic_model or DeterministicEmbeddingModel(),
        semantic_model_name=semantic_model_name,
        semantic_threshold=semantic_threshold,
        semantic_batch_size=semantic_batch_size,
        language_backend=language_backend or FixedLanguageBackend(),
        leakage_config=leakage_config,
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
    )


def test_source_loaders_preserve_raw_records_for_leakage(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [valid_row(answer="Agent response", resolution_text="Closed")],
    )

    raw_records = load_source_records(source)
    canonical_records = load_records(source)

    assert raw_records[0]["answer"] == "Agent response"
    assert raw_records[0]["resolution_text"] == "Closed"
    assert "answer" not in canonical_records[0]
    assert canonical_records[0]["message"] == "Cannot log in to my account."


def test_successful_complete_run_returns_pipeline_result(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row()])

    result = run_pipeline(source, config())

    assert isinstance(result, PipelineResult)
    assert len(result.accepted_records) == 1
    assert result.rejected_records == []
    assert result.dataset_manifest.dataset_version.startswith("ds_")
    assert result.data_quality_report.dataset_summary.accepted_records == 1


def test_mixed_records_preserve_rejections_and_do_not_reprocess_them(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    language_backend = FixedLanguageBackend()
    write_csv(
        source,
        [
            valid_row(body="Call +1 555 123 4567."),
            valid_row(body="   "),
            valid_row(body="Call +1 555 123 4567."),
            valid_row(priority=""),
            valid_row(subject="Payment problem", body="Card failed."),
        ],
    )

    result = run_pipeline(
        source,
        config(language_backend=language_backend),
    )

    assert [record.record_index for record in result.rejected_records] == [1, 3]
    assert [record.failure["reason"] for record in result.rejected_records] == [
        EMPTY_MESSAGE,
        SCHEMA_VALIDATION_FAILED,
    ]
    assert len(result.accepted_records) == 3
    assert len(language_backend.texts) == 3
    assert result.accepted_records[0].message == "Call <PHONE>."
    assert result.data_quality_report.dataset_summary.total_input_records == 5
    assert result.data_quality_report.dataset_summary.rejected_records == 2
    assert (
        result.data_quality_report.deduplication_summary.exact
        .duplicate_records_detected
        == 1
    )
    assert result.data_quality_report.pii_summary.records_with_pii == 2
    assert result.data_quality_report.pii_summary.entities_by_type == {"phone": 2}


def test_empty_dataset_completes_with_manifest_and_report(tmp_path) -> None:
    source = tmp_path / "empty.json"
    source.write_text("[]\n", encoding="utf-8")

    result = run_pipeline(source, config())

    assert result.accepted_records == []
    assert result.rejected_records == []
    assert result.dataset_manifest.counts.input == 0
    assert result.dataset_manifest.counts.processed == 0
    assert result.data_quality_report.dataset_summary.total_input_records == 0


def test_all_accepted_records_reach_final_output(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [
            valid_row(subject="Login one", body="Cannot login."),
            valid_row(subject="Login two", body="Password reset failed."),
        ],
    )

    result = run_pipeline(source, config())

    assert len(result.accepted_records) == 2
    assert result.rejected_records == []
    assert result.data_quality_report.dataset_summary.acceptance_rate == 1.0


def test_all_rejected_records_still_produce_metadata(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [
            valid_row(body="   "),
            valid_row(priority=""),
        ],
    )

    result = run_pipeline(source, config())

    assert result.accepted_records == []
    assert len(result.rejected_records) == 2
    assert result.dataset_manifest.counts.processed == 0
    assert result.data_quality_report.dataset_summary.rejection_rate == 1.0
    assert result.data_quality_report.validation_summary.issue_counts_by_reason == {
        EMPTY_MESSAGE: 1,
        SCHEMA_VALIDATION_FAILED: 1,
    }


def test_versioning_and_reporting_reuse_authoritative_stage_outputs(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row(answer="Agent response")])

    result = run_pipeline(source, config())
    expected_report = generate_data_quality_report(
        total_input_records=1,
        accepted_records=result.accepted_records,
        dataset_manifest=result.dataset_manifest,
        validation_failures=[],
        exact_duplicate_results=[],
        semantic_duplicate_results=[],
        pii_results=[],
    )

    assert result.data_quality_report.dataset_version.dataset_version == (
        result.dataset_manifest.dataset_version
    )
    assert result.data_quality_report.dataset_version.output_fingerprint == (
        result.dataset_manifest.fingerprints.output
    )
    assert result.data_quality_report.dataset_version == (
        expected_report.dataset_version
    )


def test_repeat_execution_is_deterministic_for_same_input_and_config(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row(body="Cannot log in!!!")])

    first = run_pipeline(source, config())
    second = run_pipeline(source, config())

    assert first.accepted_records == second.accepted_records
    assert first.rejected_records == second.rejected_records
    assert first.dataset_manifest == second.dataset_manifest
    assert first.data_quality_report == second.data_quality_report


def test_configuration_propagates_to_semantic_language_leakage_and_manifest(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    embedding_model = DeterministicEmbeddingModel()
    language_backend = FixedLanguageBackend(confidence=0.75)
    leakage_config = LeakageConfig(forbidden_fields={"resolution_text"})
    write_csv(source, [valid_row(resolution_text="Closed after review")])

    result = run_pipeline(
        source,
        config(
            semantic_model=embedding_model,
            language_backend=language_backend,
            semantic_threshold=0.95,
            semantic_batch_size=7,
            semantic_model_name="configured-model",
            leakage_config=leakage_config,
        ),
    )

    assert embedding_model.batch_size == 7
    assert (
        result.dataset_manifest.processing_config.semantic_deduplication
        is not None
    )
    assert (
        result.dataset_manifest.processing_config.semantic_deduplication
        .model_name
        == "configured-model"
    )
    assert (
        result.dataset_manifest.processing_config.semantic_deduplication
        .threshold
        == 0.95
    )
    assert result.accepted_records[0].language_detection is not None
    assert result.accepted_records[0].language_detection.status.value == "uncertain"
    assert result.data_quality_report.leakage_summary.issues_by_field == {
        "resolution_text": 1
    }


def test_semantic_deduplication_can_be_disabled(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row(), valid_row(body="Different issue")])

    result = run_pipeline(
        source,
        PipelineConfig(
            semantic_deduplication_enabled=False,
            language_backend=FixedLanguageBackend(),
            pipeline_version="0.1.0",
            created_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
        ),
    )

    assert result.dataset_manifest.processing_config.semantic_deduplication is None
    assert (
        result.data_quality_report.deduplication_summary.semantic
        .records_evaluated
        == 0
    )


def test_missing_source_is_fatal_not_a_rejected_record(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(PipelineExecutionError) as error:
        run_pipeline(missing, config())

    assert "loading source records" in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_invalid_semantic_configuration_is_fatal(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row()])

    with pytest.raises(PipelineExecutionError) as error:
        run_pipeline(source, config(semantic_threshold=1.5))

    assert "semantic deduplication" in str(error.value)
    assert isinstance(error.value.__cause__, ValueError)


def test_stage_order_is_explicit(monkeypatch, tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row()])
    calls: list[str] = []

    import ticket_pipeline.pipeline as pipeline_module

    def track(name, original):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        pipeline_module,
        "load_source_records",
        track("load", pipeline_module.load_source_records),
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_ticket",
        track("schema_validation", pipeline_module.validate_ticket),
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_ticket_message",
        track("message_validation", pipeline_module.validate_ticket_message),
    )
    monkeypatch.setattr(
        pipeline_module,
        "normalize_text",
        track("normalization", pipeline_module.normalize_text),
    )
    monkeypatch.setattr(
        pipeline_module,
        "mask_pii_with_metadata",
        track("pii", pipeline_module.mask_pii_with_metadata),
    )
    monkeypatch.setattr(
        pipeline_module,
        "detect_exact_duplicates",
        track("exact_dedup", pipeline_module.detect_exact_duplicates),
    )
    monkeypatch.setattr(
        pipeline_module,
        "detect_semantic_duplicate_candidates",
        track(
            "semantic_dedup",
            pipeline_module.detect_semantic_duplicate_candidates,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "annotate_ticket_language",
        track("language", pipeline_module.annotate_ticket_language),
    )
    monkeypatch.setattr(
        pipeline_module,
        "annotate_ticket_leakage",
        track("leakage", pipeline_module.annotate_ticket_leakage),
    )
    monkeypatch.setattr(
        pipeline_module,
        "create_dataset_manifest",
        track("versioning", pipeline_module.create_dataset_manifest),
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_data_quality_report",
        track("reporting", pipeline_module.generate_data_quality_report),
    )

    run_pipeline(source, config())

    assert calls == [
        "load",
        "schema_validation",
        "message_validation",
        "normalization",
        "pii",
        "normalization",
        "pii",
        "exact_dedup",
        "semantic_dedup",
        "language",
        "leakage",
        "versioning",
        "reporting",
    ]


def test_rejected_record_contract_contains_index_record_and_failure(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row(priority="")])

    result = run_pipeline(source, config())

    assert isinstance(result.rejected_records[0], RejectedRecord)
    assert result.rejected_records[0].record_index == 0
    assert result.rejected_records[0].canonical_record["message"] == (
        "Cannot log in to my account."
    )
    assert result.rejected_records[0].failure["reason"] == (
        SCHEMA_VALIDATION_FAILED
    )


def test_dataset_version_is_stable_despite_backend_confidence_noise(
    tmp_path,
) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [
            valid_row(subject="Login one", body="Cannot log in."),
            valid_row(subject="Login two", body="Password reset failed."),
        ],
    )
    noisy_backend = NoisyLanguageBackend()

    first = run_pipeline(source, config(language_backend=noisy_backend))
    second = run_pipeline(source, config(language_backend=noisy_backend))

    assert noisy_backend.calls == 4
    assert first.accepted_records == second.accepted_records
    assert first.dataset_manifest.dataset_version == (
        second.dataset_manifest.dataset_version
    )
    assert first.dataset_manifest.fingerprints.output == (
        second.dataset_manifest.fingerprints.output
    )


def test_json_source_runs_through_every_stage(tmp_path) -> None:
    source = tmp_path / "tickets.json"
    source.write_text(
        json.dumps(
            [
                valid_row(subject="Login issue", body="Cannot log in."),
                valid_row(body="   "),
                valid_row(subject="Payment problem", body="Card failed."),
            ]
        ),
        encoding="utf-8",
    )

    result = run_pipeline(source, config())

    assert [record.record_index for record in result.rejected_records] == [1]
    assert result.rejected_records[0].failure["reason"] == EMPTY_MESSAGE
    assert len(result.accepted_records) == 2
    assert result.data_quality_report.dataset_summary.total_input_records == 3
    assert result.dataset_manifest.counts.processed == 2
    assert result.accepted_records[0].language_detection is not None
    assert result.accepted_records[0].leakage_check is not None


def test_default_config_without_semantic_dependency_fails_fast(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(source, [valid_row()])

    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    with pytest.raises(PipelineExecutionError) as error:
        run_pipeline(source, PipelineConfig())

    assert "semantic_deduplication_enabled=False" in str(error.value)
