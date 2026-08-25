"""Schema-validate every canonical record from the real raw dataset.

This is the reproducible counterpart to the manual dataset check performed
during review: it proves (on every test run, not just once ad hoc) that all
28,587 records in data/raw/tickets_raw.csv are schema-valid.
"""

from collections import Counter

from ticket_pipeline.loaders import load_csv
from ticket_pipeline.validators import validate_ticket


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
