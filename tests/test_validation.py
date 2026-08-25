from ticket_pipeline.models import Ticket
from ticket_pipeline.validators import SCHEMA_VALIDATION_FAILED, validate_ticket


def canonical_record() -> dict[str, object]:
    return {
        "subject": None,
        "message": "Payment failed",
        "ticket_type": "Incident",
        "queue": "Billing",
        "priority": "high",
        "language": "en",
        "source_version": 52,
        "tags": [],
    }


def assert_failed_at(record: dict[str, object], field: str, error_type: str) -> None:
    ticket, failure = validate_ticket(record)

    assert ticket is None
    assert failure is not None
    assert failure["reason"] == SCHEMA_VALIDATION_FAILED
    assert any(
        error["loc"] == (field,) and error["type"] == error_type
        for error in failure["errors"]
    )


def test_valid_ticket_passes_schema_validation() -> None:
    ticket, failure = validate_ticket(canonical_record())

    assert isinstance(ticket, Ticket)
    assert ticket.message == "Payment failed"
    assert failure is None


def test_missing_required_message_fails_with_pydantic_details() -> None:
    record = canonical_record()
    del record["message"]

    assert_failed_at(record, "message", "missing")


def test_invalid_priority_fails_schema_validation() -> None:
    record = canonical_record()
    record["priority"] = "urgent"

    assert_failed_at(record, "priority", "literal_error")


def test_nullable_subject_passes_schema_validation() -> None:
    ticket, failure = validate_ticket(canonical_record())

    assert ticket is not None
    assert ticket.subject is None
    assert failure is None


def test_omitted_subject_uses_model_default() -> None:
    record = canonical_record()
    del record["subject"]

    ticket, failure = validate_ticket(record)

    assert ticket is not None
    assert ticket.subject is None
    assert failure is None


def test_string_tags_fail_schema_validation() -> None:
    record = canonical_record()
    record["tags"] = "billing"

    assert_failed_at(record, "tags", "list_type")


def test_non_string_tag_fails_schema_validation() -> None:
    record = canonical_record()
    record["tags"] = ["billing", 123]

    ticket, failure = validate_ticket(record)

    assert ticket is None
    assert failure is not None
    assert any(
        error["loc"] == ("tags", 1) and error["type"] == "string_type"
        for error in failure["errors"]
    )


def test_invalid_source_version_fails_schema_validation() -> None:
    record = canonical_record()
    record["source_version"] = "abc"

    assert_failed_at(record, "source_version", "int_parsing")


def test_simple_source_version_string_is_intentionally_coerced() -> None:
    record = canonical_record()
    record["source_version"] = "51"

    ticket, failure = validate_ticket(record)

    assert ticket is not None
    assert ticket.source_version == 51
    assert failure is None


def test_wrong_message_type_fails_schema_validation() -> None:
    record = canonical_record()
    record["message"] = ["Payment failed"]

    assert_failed_at(record, "message", "string_type")


def test_whitespace_only_message_remains_schema_valid() -> None:
    record = canonical_record()
    record["message"] = "   "

    ticket, failure = validate_ticket(record)

    assert ticket is not None
    assert ticket.message == "   "
    assert failure is None
