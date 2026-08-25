"""Structural validation for canonical support-ticket records."""

from typing import Literal, TypedDict

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from ticket_pipeline.models import Ticket


SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"


class SchemaValidationFailure(TypedDict):
    """Traceable details for a canonical record that fails validation."""

    reason: Literal["SCHEMA_VALIDATION_FAILED"]
    errors: list[ErrorDetails]


def validate_ticket(
    record: dict[str, object],
) -> tuple[Ticket | None, SchemaValidationFailure | None]:
    """Validate a canonical dictionary without raising for record-level errors.

    Exactly one tuple item is populated: a validated ``Ticket`` on success, or
    a failure containing Pydantic's structured error details on failure.
    """

    try:
        return Ticket.model_validate(record), None
    except ValidationError as error:
        return None, {
            "reason": SCHEMA_VALIDATION_FAILED,
            "errors": error.errors(),
        }
