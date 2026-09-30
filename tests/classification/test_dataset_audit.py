import json
from datetime import datetime, timezone

import numpy as np
import pytest

from ticket_classification.audit import audit_classification_dataset
from ticket_classification.dataset import (
    ClassificationAuditConfig,
    build_classification_records,
    build_classification_text,
    build_record_ids,
    load_accepted_records,
)
from ticket_classification.models import DatasetReadinessStatus, FieldUsage
from ticket_classification.outputs import (
    classification_audit_to_json,
    classification_audit_to_markdown,
    write_classification_audit_outputs,
)
from ticket_classification.workflow import run_classification_dataset_audit
from ticket_pipeline.language import LanguagePrediction
from ticket_pipeline.models import DatasetSourceManifest, SourceFileManifest, Ticket
from ticket_pipeline.outputs import write_pipeline_outputs
from ticket_pipeline.pipeline import PipelineConfig, PipelineResult, run_pipeline
from ticket_pipeline.versioning import (
    capture_processing_config,
    create_dataset_manifest,
    load_dataset_manifest,
)


class DeterministicEmbeddingModel:
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


def manifest(records: list[Ticket]):
    return create_dataset_manifest(
        source=DatasetSourceManifest(
            files=[
                SourceFileManifest(
                    name="accepted.jsonl",
                    sha256="a" * 64,
                    size_bytes=100,
                )
            ]
        ),
        processing_config=capture_processing_config(
            semantic_deduplication_enabled=False
        ),
        input_record_count=len(records),
        processed_records=records,
        pipeline_version="0.1.0",
        created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
    )


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
    ]
    lines = [",".join(fields)]
    for row in rows:
        values = [row.get(field, "") for field in fields]
        lines.append(",".join(f'"{value}"' for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(**updates: str) -> dict[str, str]:
    values = {
        "subject": "Login issue",
        "body": "Cannot log in to my account.",
        "answer": "",
        "type": "Incident",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en",
        "version": "52",
        "tag_1": "login",
    }
    values.update(updates)
    return values


def test_classification_text_uses_subject_and_message_deterministically() -> None:
    record = ticket(subject="  Login issue  ", message=" Cannot log in. ")

    assert build_classification_text(record, ("subject", "message")) == (
        "Login issue\n\nCannot log in."
    )
    assert build_classification_text(
        ticket(subject=None, message="Cannot log in."),
        ("subject", "message"),
    ) == "Cannot log in."
    assert build_classification_text(
        ticket(subject="Login issue", message="   "),
        ("subject", "message"),
    ) == "Login issue"


def test_default_audit_detects_single_label_multiclass_contract() -> None:
    audit = audit_classification_dataset(
        [
            ticket(ticket_type="Incident"),
            ticket(ticket_type="Request"),
            ticket(ticket_type="Problem"),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1),
        dataset_manifest=manifest([]),
    )

    assert audit.classification_task.target_field == "ticket_type"
    assert audit.classification_task.task_type == "single_label_multiclass"
    assert audit.classification_task.input_fields == ["subject", "message"]
    assert audit.dataset_summary.dataset_version is not None


def test_binary_one_class_and_multilabel_task_type_detection() -> None:
    binary = audit_classification_dataset(
        [ticket(priority="low"), ticket(priority="high")],
        ClassificationAuditConfig(target_field="priority", rare_class_min_samples=1),
    )
    one_class = audit_classification_dataset(
        [ticket(ticket_type="Incident"), ticket(ticket_type="Incident")],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )
    multilabel = audit_classification_dataset(
        [ticket(tags=["billing", "refund"]), ticket(tags=["login"])],
        ClassificationAuditConfig(target_field="tags", rare_class_min_samples=1),
    )

    assert binary.classification_task.task_type == "binary"
    assert one_class.classification_task.task_type == "not_supervised"
    assert one_class.dataset_readiness is DatasetReadinessStatus.NOT_READY
    assert multilabel.classification_task.task_type == "multilabel"


def test_class_counts_percentages_and_imbalance_are_reported() -> None:
    audit = audit_classification_dataset(
        [
            ticket(ticket_type="Incident"),
            ticket(ticket_type="Incident"),
            ticket(ticket_type="Incident"),
            ticket(ticket_type="Request"),
            ticket(ticket_type="Problem"),
        ],
        ClassificationAuditConfig(rare_class_min_samples=2),
    )

    assert [(item.class_name, item.count, item.percentage) for item in audit.class_distribution.classes] == [
        ("Incident", 3, 0.6),
        ("Problem", 1, 0.2),
        ("Request", 1, 0.2),
    ]
    assert audit.class_distribution.largest_class == "Incident"
    assert audit.class_distribution.smallest_class == "Request"
    assert audit.class_distribution.largest_to_smallest_ratio == 3.0
    assert audit.class_distribution.imbalance_severity == "mild"
    assert [item.class_name for item in audit.rare_classes] == ["Problem", "Request"]


def test_missing_blank_invalid_and_real_unknown_labels_are_distinct() -> None:
    invalid_target_ticket = ticket()
    audit = audit_classification_dataset(
        [
            ticket(subject=None),
            ticket(subject=""),
            ticket(subject="Unknown"),
            invalid_target_ticket,
        ],
        ClassificationAuditConfig(
            target_field="subject",
            input_fields=("message",),
            rare_class_min_samples=1,
        ),
    )
    invalid = audit_classification_dataset(
        [invalid_target_ticket],
        ClassificationAuditConfig(
            target_field="source_version",
            input_fields=("subject", "message"),
        ),
    )

    assert audit.missing_target_summary.missing_target_records == 1
    assert audit.missing_target_summary.blank_target_records == 1
    assert "Unknown" in audit.target_summary.class_names
    assert invalid.missing_target_summary.invalid_target_records == 1
    # An all-invalid-target dataset must still be NOT_READY after removing
    # the redundant `invalid_target_records == total_records` readiness
    # clause: `records_with_usable_labels == 0` already covers this case.
    assert invalid.dataset_readiness is DatasetReadinessStatus.NOT_READY


def test_list_target_whitespace_only_items_are_blank_not_invalid() -> None:
    audit = audit_classification_dataset(
        [
            ticket(tags=[]),
            ticket(tags=[""]),
            ticket(tags=["   "]),
        ],
        ClassificationAuditConfig(
            target_field="tags",
            input_fields=("subject", "message"),
            rare_class_min_samples=1,
        ),
    )

    assert audit.missing_target_summary.blank_target_records == 3
    assert audit.missing_target_summary.invalid_target_records == 0
    assert audit.missing_target_summary.valid_target_records == 0

    valid = audit_classification_dataset(
        [ticket(tags=["Billing", "Refund"])],
        ClassificationAuditConfig(
            target_field="tags",
            input_fields=("subject", "message"),
            rare_class_min_samples=1,
        ),
    )

    assert valid.missing_target_summary.valid_target_records == 1
    assert valid.target_summary.class_names == ["Billing", "Refund"]


def test_suspicious_label_formatting_is_reported_without_merging() -> None:
    audit = audit_classification_dataset(
        [
            ticket(ticket_type="Billing"),
            ticket(ticket_type="billing"),
            ticket(ticket_type=" BILLING "),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )

    assert audit.target_summary.unique_class_count == 3
    assert audit.target_summary.suspicious_label_groups == [
        [" BILLING ", "Billing", "billing"]
    ]


def test_text_length_and_short_placeholder_inputs_are_reported() -> None:
    audit = audit_classification_dataset(
        [
            ticket(subject=None, message="<EMAIL>"),
            ticket(subject=None, message="OK"),
            ticket(subject="Long", message="word " * 100),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1, short_text_min_words=3),
    )

    assert audit.text_length_summary.word_length.minimum == 1
    assert audit.text_length_summary.word_length.maximum == 101
    assert audit.text_length_summary.word_length.median == 1
    assert audit.text_length_summary.short_input_records == 2
    assert audit.text_length_summary.placeholder_only_records == 1


def test_empty_dataset_and_all_missing_labels_are_not_ready() -> None:
    empty = audit_classification_dataset([])
    all_missing = audit_classification_dataset(
        [ticket(subject=None), ticket(subject=None)],
        ClassificationAuditConfig(target_field="subject", input_fields=("message",)),
    )

    assert empty.dataset_readiness is DatasetReadinessStatus.NOT_READY
    assert empty.classification_task.task_type == "not_supervised"
    assert all_missing.dataset_readiness is DatasetReadinessStatus.NOT_READY
    assert all_missing.target_summary.records_with_missing_labels == 2


def test_duplicate_inputs_with_same_and_conflicting_labels_are_reported_safely() -> None:
    conflict_a = ticket(subject="Conflict", message="Same text", ticket_type="Request")
    conflict_b = ticket(subject="Conflict", message="Same text", ticket_type="Problem")
    audit = audit_classification_dataset(
        [
            ticket(subject="Same", message="Same text", ticket_type="Incident"),
            ticket(subject="Same", message="Same text", ticket_type="Incident"),
            conflict_a,
            conflict_b,
        ],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )
    expected_ids = sorted(build_record_ids([conflict_a, conflict_b]))

    assert audit.conflicting_labels.duplicated_input_groups == 2
    assert audit.conflicting_labels.groups_with_consistent_labels == 1
    assert audit.conflicting_labels.groups_with_conflicting_labels == 1
    example = audit.conflicting_labels.conflicting_examples[0]
    assert sorted(example["record_ids"]) == expected_ids
    assert example["labels"] == ["Problem", "Request"]
    assert "Same text" not in json.dumps(example)


def test_input_field_policy_excludes_metadata_and_leakage_risks() -> None:
    audit = audit_classification_dataset(
        [
            ticket(ticket_type="Incident"),
            ticket(ticket_type="Request"),
            ticket(ticket_type="Problem"),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )
    policy = {item.field_name: item.usage for item in audit.input_field_policy}

    assert policy["subject"] is FieldUsage.ALLOWED_INPUT
    assert policy["message"] is FieldUsage.ALLOWED_INPUT
    assert policy["ticket_type"] is FieldUsage.TARGET
    assert policy["queue"] is FieldUsage.LEAKAGE_RISK
    assert policy["priority"] is FieldUsage.LEAKAGE_RISK
    assert policy["tags"] is FieldUsage.LEAKAGE_RISK
    assert policy["source_version"] is FieldUsage.IDENTIFIER
    assert policy["language_detection"] is FieldUsage.NOT_AVAILABLE_AT_INFERENCE


def test_readiness_ready_ready_with_warnings_and_not_ready() -> None:
    ready = audit_classification_dataset(
        [
            ticket(subject="A1", message="Alpha one", ticket_type="A"),
            ticket(subject="A2", message="Alpha two", ticket_type="A"),
            ticket(subject="B1", message="Beta one", ticket_type="B"),
            ticket(subject="B2", message="Beta two", ticket_type="B"),
            ticket(subject="C1", message="Gamma one", ticket_type="C"),
            ticket(subject="C2", message="Gamma two", ticket_type="C"),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1, short_text_min_words=1),
    )
    with_warnings = audit_classification_dataset(
        [ticket(ticket_type="A"), ticket(ticket_type="B"), ticket(ticket_type="C")],
        ClassificationAuditConfig(rare_class_min_samples=2),
    )
    not_ready = audit_classification_dataset(
        [ticket(ticket_type="A"), ticket(ticket_type="A")],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )

    assert ready.dataset_readiness is DatasetReadinessStatus.READY
    assert ready.warnings == []
    assert ready.leakage_risks
    assert with_warnings.dataset_readiness is DatasetReadinessStatus.READY_WITH_WARNINGS
    assert not_ready.dataset_readiness is DatasetReadinessStatus.NOT_READY


def test_classification_records_include_id_text_label_and_source_fields() -> None:
    incident = ticket(ticket_type="Incident")
    request = ticket(ticket_type="Request")
    records = build_classification_records([incident, request])
    expected_ids = build_record_ids([incident, request])

    assert records[0].record_id == expected_ids[0]
    assert records[1].record_id == expected_ids[1]
    assert records[0].label == "Incident"
    assert records[0].source_fields == ["subject", "message"]
    assert records[0].text == "Login issue\n\nCannot log in to my account."


def test_content_derived_record_ids_are_pure_functions_of_record_content() -> None:
    original = ticket(ticket_type="Incident")
    same_content = ticket(ticket_type="Incident")
    different_priority = ticket(ticket_type="Incident", priority="low")

    original_id = build_record_ids([original])[0]
    same_content_id = build_record_ids([same_content])[0]
    different_id = build_record_ids([different_priority])[0]

    assert original_id.startswith("record:")
    assert original_id == same_content_id
    assert original_id != different_id


def test_content_derived_record_ids_are_stable_across_reordering() -> None:
    incident = ticket(ticket_type="Incident")
    request = ticket(ticket_type="Request")

    forward = build_record_ids([incident, request])
    backward = build_record_ids([request, incident])

    assert forward[0] == backward[1]
    assert forward[1] == backward[0]


def test_content_derived_record_ids_disambiguate_true_duplicates() -> None:
    first = ticket(ticket_type="Incident")
    duplicate = ticket(ticket_type="Incident")

    ids = build_record_ids([first, duplicate])

    assert ids[0] != ids[1]
    assert ids[1] == f"{ids[0]}:1"


def test_deterministic_json_and_markdown_outputs(tmp_path) -> None:
    audit = audit_classification_dataset(
        [ticket(ticket_type="Incident"), ticket(ticket_type="Request")],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )

    first_json = classification_audit_to_json(audit)
    second_json = classification_audit_to_json(audit)
    markdown = classification_audit_to_markdown(audit)
    artifacts = write_classification_audit_outputs(audit, tmp_path)

    assert first_json == second_json
    assert json.loads(artifacts.json_path.read_text(encoding="utf-8")) == (
        audit.model_dump(mode="json")
    )
    assert "Classification Dataset Audit" in markdown
    assert artifacts.markdown_path.read_text(encoding="utf-8") == markdown


def test_unicode_input_survives_audit_and_json_output_without_corruption() -> None:
    # The audit never echoes raw ticket text (subject/message) into its
    # output by design (see the conflicting-label privacy test above), so
    # unicode integrity is exercised through the target label, which the
    # audit does surface via class names and class distribution.
    audit = audit_classification_dataset(
        [
            ticket(
                subject="Problème de paiement 支払い",
                message="Impossible de se connecter 😀 café",
                ticket_type="退款",
            ),
            ticket(
                subject="退款请求",
                message="需要退款",
                ticket_type="Emoji 😀 Label",
            ),
        ],
        ClassificationAuditConfig(rare_class_min_samples=1),
    )

    payload = classification_audit_to_json(audit)
    reloaded = json.loads(payload)

    assert "退款" in payload
    assert "😀" in payload
    assert "\\u" not in payload
    assert set(audit.target_summary.class_names) == {"退款", "Emoji 😀 Label"}
    assert audit.dataset_readiness is not DatasetReadinessStatus.NOT_READY
    assert reloaded["dataset_readiness"] == audit.dataset_readiness.value


def test_accepted_jsonl_loader_validates_phase1_records(tmp_path) -> None:
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        "\n".join(
            json.dumps(record.model_dump(mode="json"))
            for record in [
                ticket(ticket_type="Incident"),
                ticket(ticket_type="Request"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_accepted_records(accepted_path)

    assert [record.ticket_type for record in records] == ["Incident", "Request"]


def test_missing_manifest_path_fails_clearly_instead_of_silently_ignoring(tmp_path) -> None:
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        json.dumps(ticket(ticket_type="Incident").model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    missing_manifest_path = tmp_path / "does_not_exist_manifest.json"

    with pytest.raises(FileNotFoundError):
        run_classification_dataset_audit(
            accepted_path,
            manifest_path=missing_manifest_path,
        )


def test_integration_consumes_phase1_accepted_artifact_and_persists_audit(tmp_path) -> None:
    source = tmp_path / "tickets.csv"
    write_csv(
        source,
        [
            row(type="Incident", body="Cannot log in."),
            row(type="Request", subject="Access", body="Need new access."),
            row(type="Problem", subject="Payment", body="Card failed."),
        ],
    )
    result = run_pipeline(
        source,
        PipelineConfig(
            semantic_model=DeterministicEmbeddingModel(),
            semantic_model_name="test-classification-embedding-v1",
            language_backend=EnglishBackend(),
            pipeline_version="0.1.0",
            created_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        ),
    )
    ingestion_artifacts = write_pipeline_outputs(result, tmp_path / "phase1")

    audit, audit_artifacts = run_classification_dataset_audit(
        ingestion_artifacts.accepted_path,
        manifest_path=ingestion_artifacts.manifest_path,
        output_dir=tmp_path / "classification",
        config=ClassificationAuditConfig(rare_class_min_samples=1),
    )

    loaded_manifest = load_dataset_manifest(ingestion_artifacts.manifest_path)
    assert audit.dataset_summary.total_records == 3
    assert audit.dataset_summary.usable_labeled_records == 3
    assert audit.dataset_summary.dataset_version == loaded_manifest.dataset_version
    assert audit.classification_task.task_type == "single_label_multiclass"
    assert audit_artifacts.json_path.name == "classification_dataset_audit.json"
    assert audit_artifacts.markdown_path.name == "classification_dataset_audit.md"


def test_unknown_configured_fields_fail_fast() -> None:
    with pytest.raises(ValueError, match="unknown Ticket field"):
        audit_classification_dataset(
            [ticket()],
            ClassificationAuditConfig(target_field="not_a_field"),
        )

    with pytest.raises(ValueError, match="unsupported classification input field"):
        audit_classification_dataset(
            [ticket()],
            ClassificationAuditConfig(input_fields=("queue",)),
        )
