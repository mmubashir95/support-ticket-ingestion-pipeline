"""Schema and message-usability validation for support tickets."""

import unicodedata
from html.parser import HTMLParser
from typing import Literal, TypedDict

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from ticket_pipeline.models import Ticket


SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
EMPTY_MESSAGE = "EMPTY_MESSAGE"


class SchemaValidationFailure(TypedDict):
    """Traceable details for a canonical record that fails validation."""

    reason: Literal["SCHEMA_VALIDATION_FAILED"]
    errors: list[ErrorDetails]


class MessageValidationFailure(TypedDict):
    """Reason a schema-valid ticket is unusable for text processing."""

    reason: Literal["EMPTY_MESSAGE"]


class _VisibleTextParser(HTMLParser):
    """Collect text content while ignoring markup and comments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


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


def is_message_empty(message: str | None) -> bool:
    """Return whether a message has no visible text content.

    HTML and control characters are ignored only for this check. The input
    message is never rewritten or normalized.
    """

    if message is None or not message.strip():
        return True

    parser = _VisibleTextParser()
    parser.feed(message)
    parser.close()

    visible_text = "".join(parser.parts)
    visible_text = "".join(
        character
        for character in visible_text
        if unicodedata.category(character) != "Cc"
    )
    return not visible_text.strip()


def validate_ticket_message(
    ticket: Ticket,
) -> tuple[Ticket | None, MessageValidationFailure | None]:
    """Validate that a schema-valid ticket has usable visible message content."""

    if is_message_empty(ticket.message):
        return None, {"reason": EMPTY_MESSAGE}
    return ticket, None
