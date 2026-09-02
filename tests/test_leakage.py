import logging

import pytest
from pydantic import ValidationError

from ticket_pipeline.language import (
    LanguageDetector,
    LanguagePrediction,
    annotate_ticket_language,
)
from ticket_pipeline.leakage import (
    LeakageChecker,
    LeakageConfig,
    annotate_ticket_leakage,
    is_populated,
)
from ticket_pipeline.models import (
    LanguageDetectionStatus,
    LeakageCheckMetadata,
    LeakageCheckStatus,
    LeakageIssue,
    LeakageIssueType,
    Ticket,
)


class EnglishBackend:
    def predict(self, text: str) -> LanguagePrediction:
        return LanguagePrediction("en", 0.98)


def make_ticket() -> Ticket:
    ticket = Ticket(
        subject="Refund not received",
        message="I requested a refund yesterday but it still has not arrived.",
        ticket_type="Incident",
        queue="Billing and Payments",
        priority="high",
        language="en",
        source_version=52,
        tags=["refund"],
    )
    return annotate_ticket_language(
        ticket,
        LanguageDetector(backend=EnglishBackend()),
    )


def exact_result(
    record_index: int,
    duplicate_of: int | None,
) -> dict[str, object]:
    return {
        "record_index": record_index,
        "is_exact_duplicate": duplicate_of is not None,
        "duplicate_of": duplicate_of,
        "deduplication_fingerprint": "a" * 64,
    }


def semantic_result(
    record_index: int,
    duplicate_of: int | None,
) -> dict[str, object]:
    return {
        "record_index": record_index,
        "is_semantic_duplicate_candidate": duplicate_of is not None,
        "candidate_duplicate_of": duplicate_of,
        "nearest_earlier_record": duplicate_of,
        "semantic_similarity": 0.94 if duplicate_of is not None else None,
        "threshold": 0.85,
        "skipped_exact_duplicate": False,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("   ", False),
        ([], False),
        ({}, False),
        (False, True),
        (0, True),
        ("value", True),
        (["value"], True),
        ({"key": "value"}, True),
    ],
)
def test_populated_value_policy(value: object, expected: bool) -> None:
    assert is_populated(value) is expected


def test_clean_ticket_has_clean_status() -> None:
    result = LeakageChecker().check({"subject": "Refund not received"})

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.issues == []
    assert result.grouping.identifiers == {}


def test_default_policy_flags_populated_source_answer() -> None:
    result = LeakageChecker().check(
        {"answer": "The refund was completed by the billing agent."}
    )

    assert result.status is LeakageCheckStatus.WARNING
    assert result.issues == [
        LeakageIssue(type=LeakageIssueType.FUTURE_FIELD, field="answer")
    ]


@pytest.mark.parametrize("value", [None, "", "   ", [], {}])
def test_empty_forbidden_field_is_not_flagged(value: object) -> None:
    checker = LeakageChecker(
        LeakageConfig(forbidden_fields={"closed_at"})
    )

    result = checker.check({"closed_at": value})

    assert result.status is LeakageCheckStatus.CLEAN


@pytest.mark.parametrize("value", [False, 0])
def test_false_and_zero_are_populated_future_values(value: object) -> None:
    checker = LeakageChecker(
        LeakageConfig(forbidden_fields={"refund_issued"})
    )

    result = checker.check({"refund_issued": value})

    assert result.status is LeakageCheckStatus.WARNING
    assert result.issues[0].type is LeakageIssueType.FUTURE_FIELD


def test_configured_target_is_informational_not_a_warning() -> None:
    """A populated target field is a normal labeled row, not leakage.

    Presence of the label a record is meant to train on does not mean that
    label leaked into a model's input features, so it must not raise the
    same status as a genuinely forbidden post-outcome field.
    """
    checker = LeakageChecker(
        LeakageConfig(forbidden_fields=set(), target_fields={"priority"})
    )

    result = checker.check({"priority": "high"})

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.issues == []
    assert result.targets.target_fields == ["priority"]


def test_configured_target_and_proxy_are_informational() -> None:
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields=set(),
            target_fields={"needs_escalation"},
            target_proxy_fields={"assigned_team"},
        )
    )

    result = checker.check(
        {"needs_escalation": False, "assigned_team": "L3 Escalation"}
    )

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.issues == []
    assert result.targets.target_fields == ["needs_escalation"]
    assert result.targets.target_proxy_fields == ["assigned_team"]


def test_no_target_configured_skips_proxy_check() -> None:
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields=set(),
            target_proxy_fields={"assigned_team"},
        )
    )

    result = checker.check({"assigned_team": "L3 Escalation"})

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.issues == []
    assert result.targets.target_fields == []
    assert result.targets.target_proxy_fields == []


def test_missing_configured_fields_do_not_crash() -> None:
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields={"closed_at", "resolved_at"},
            target_fields={"needs_escalation"},
            target_proxy_fields={"assigned_team"},
            group_fields={"conversation_id"},
        )
    )

    result = checker.check({"subject": "Refund not received"})

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.unchecked_fields == [
        "assigned_team",
        "closed_at",
        "conversation_id",
        "needs_escalation",
        "resolved_at",
    ]


def test_omitting_source_fields_leaves_answer_unchecked_not_silently_clean() -> (
    None
):
    """Forgetting source_fields must not look identical to a checked, clean record.

    The default answer field only ever appears in the merged record when a
    caller passes source_fields, since answer is intentionally excluded from
    the canonical Ticket. A future orchestrator that forgets to pass
    source_fields would otherwise get status=clean indistinguishable from a
    record that was actually checked and found free of the default forbidden
    field. unchecked_fields makes that gap visible instead of silent.
    """
    ticket = make_ticket()
    checker = LeakageChecker()

    annotated = annotate_ticket_leakage(ticket, checker)

    assert annotated.leakage_check is not None
    assert annotated.leakage_check.status is LeakageCheckStatus.CLEAN
    assert annotated.leakage_check.issues == []
    assert annotated.leakage_check.unchecked_fields == ["answer"]


def test_exact_duplicate_link_becomes_informational_group() -> None:
    result = LeakageChecker().check(
        {},
        exact_duplicate=exact_result(4, 1),  # type: ignore[arg-type]
    )

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.grouping.exact_duplicate_group_id == 1
    assert result.issues == []


def test_semantic_duplicate_link_becomes_informational_group() -> None:
    result = LeakageChecker().check(
        {},
        semantic_duplicate=semantic_result(5, 2),  # type: ignore[arg-type]
    )

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.grouping.semantic_duplicate_group_id == 2


def test_group_identifiers_are_exposed_without_warning() -> None:
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields=set(),
            group_fields={"conversation_id", "thread_id", "customer_id"},
        )
    )

    result = checker.check(
        {
            "conversation_id": "CONV-123",
            "thread_id": 42,
            "customer_id": "CUSTOMER-9",
        }
    )

    assert result.status is LeakageCheckStatus.CLEAN
    assert result.grouping.identifiers == {
        "conversation_id": "CONV-123",
        "customer_id": "CUSTOMER-9",
        "thread_id": 42,
    }


def test_blank_group_identifier_is_ignored() -> None:
    checker = LeakageChecker(
        LeakageConfig(forbidden_fields=set(), group_fields={"thread_id"})
    )

    result = checker.check({"thread_id": "  "})

    assert result.grouping.identifiers == {}


def test_annotation_preserves_ticket_text_and_previous_metadata() -> None:
    ticket = make_ticket()
    checker = LeakageChecker(
        LeakageConfig(
            forbidden_fields={"closed_at", "resolution_text"},
            group_fields={"conversation_id"},
        )
    )

    annotated = annotate_ticket_leakage(
        ticket,
        checker,
        source_fields={
            "closed_at": "2026-09-03T09:00:00Z",
            "resolution_text": "Refund completed",
            "conversation_id": "CONV-123",
        },
    )

    assert annotated is not ticket
    assert annotated.subject == ticket.subject
    assert annotated.message == ticket.message
    assert annotated.language_detection == ticket.language_detection
    assert ticket.leakage_check is None
    assert annotated.leakage_check is not None
    assert annotated.leakage_check.status is LeakageCheckStatus.WARNING
    assert [issue.field for issue in annotated.leakage_check.issues] == [
        "closed_at",
        "resolution_text",
    ]
    assert annotated.leakage_check.grouping.identifiers == {
        "conversation_id": "CONV-123"
    }
    assert annotated.leakage_check.unchecked_fields == []


def test_canonical_ticket_fields_can_be_checked_as_targets() -> None:
    ticket = make_ticket()
    checker = LeakageChecker(
        LeakageConfig(forbidden_fields=set(), target_fields={"priority"})
    )

    annotated = annotate_ticket_leakage(ticket, checker)

    assert annotated.leakage_check is not None
    assert annotated.leakage_check.status is LeakageCheckStatus.CLEAN
    assert annotated.leakage_check.issues == []
    assert annotated.leakage_check.targets.target_fields == ["priority"]


def test_invalid_duplicate_metadata_fails_safely_and_logs_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ticket_text = "I requested a refund yesterday but it still has not arrived."
    invalid_result = exact_result(1, 0)
    invalid_result["duplicate_of"] = 2
    checker = LeakageChecker()

    with caplog.at_level(logging.ERROR):
        result = checker.check(
            {"message": ticket_text},
            exact_duplicate=invalid_result,  # type: ignore[arg-type]
        )

    assert result.status is LeakageCheckStatus.FAILED
    assert result.issues == []
    assert "Leakage check failed" in caplog.text
    assert ticket_text not in caplog.text


def test_misaligned_duplicate_results_fail_safely() -> None:
    result = LeakageChecker().check(
        {},
        exact_duplicate=exact_result(1, 0),  # type: ignore[arg-type]
        semantic_duplicate=semantic_result(2, None),  # type: ignore[arg-type]
    )

    assert result.status is LeakageCheckStatus.FAILED


@pytest.mark.parametrize(
    "config",
    [
        {"forbidden_fields": "answer"},
        {"target_fields": {""}},
        {"target_proxy_fields": {1}},
        {"group_fields": {"   "}},
    ],
)
def test_invalid_configuration_is_rejected(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LeakageConfig(**config)  # type: ignore[arg-type]


def test_leakage_metadata_status_contract_is_validated() -> None:
    with pytest.raises(ValidationError):
        LeakageCheckMetadata(
            status=LeakageCheckStatus.CLEAN,
            issues=[
                LeakageIssue(
                    type=LeakageIssueType.FUTURE_FIELD,
                    field="closed_at",
                )
            ],
        )


def test_language_metadata_fixture_is_detected() -> None:
    ticket = make_ticket()

    assert ticket.language_detection is not None
    assert ticket.language_detection.status is LanguageDetectionStatus.DETECTED
