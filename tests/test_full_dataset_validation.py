"""Run every canonical record from the real raw dataset through the full
schema validation -> business validation -> normalization pipeline.

This is the reproducible counterpart to the manual dataset checks performed
during review: it proves (on every test run, not just once ad hoc) that all
28,587 records in data/raw/tickets_raw.csv are schema-valid, that their
messages are business-valid, and that normalization succeeds and is
idempotent for every one of them.
"""

from collections import Counter

from ticket_pipeline.loaders import load_csv
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.validators import validate_ticket, validate_ticket_message


EXPECTED_RECORD_COUNT = 28_587


def test_full_raw_dataset_is_schema_valid() -> None:
    records = load_csv("data/raw/tickets_raw.csv")
    assert len(records) == EXPECTED_RECORD_COUNT

    invalid_error_types: Counter[tuple[str, ...]] = Counter()
    invalid_count = 0

    for record in records:
        ticket, failure = validate_ticket(record)
        if ticket is None:
            invalid_count += 1
            assert failure is not None
            for error in failure["errors"]:
                invalid_error_types[(str(error["loc"]), error["type"])] += 1

    assert invalid_count == 0, (
        f"{invalid_count} of {len(records)} records failed schema validation: "
        f"{invalid_error_types.most_common(10)}"
    )


def test_full_raw_dataset_messages_are_business_valid_and_normalize_safely() -> None:
    records = load_csv("data/raw/tickets_raw.csv")

    schema_valid = 0
    empty_message_count = 0
    normalization_failures: list[str] = []
    non_idempotent_count = 0

    for record in records:
        ticket, schema_failure = validate_ticket(record)
        if ticket is None:
            continue
        schema_valid += 1

        validated, business_failure = validate_ticket_message(ticket)
        if validated is None:
            empty_message_count += 1
            continue

        try:
            normalized_once = normalize_text(validated.message)
            normalized_twice = normalize_text(normalized_once)
        except Exception as error:  # noqa: BLE001 - collect and report, don't fail fast
            normalization_failures.append(f"{validated.message[:50]!r}: {error}")
            continue

        if normalized_once != normalized_twice:
            non_idempotent_count += 1

    assert schema_valid == EXPECTED_RECORD_COUNT
    assert not normalization_failures, (
        f"{len(normalization_failures)} messages raised during normalization: "
        f"{normalization_failures[:5]}"
    )
    assert non_idempotent_count == 0, (
        f"{non_idempotent_count} messages were not idempotent under normalize_text"
    )
