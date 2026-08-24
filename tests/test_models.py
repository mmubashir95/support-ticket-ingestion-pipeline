import pytest
from pydantic import ValidationError

from ticket_pipeline.models import Ticket


def test_valid_ticket_can_be_created() -> None:
    ticket = Ticket(
        subject="Account Disruption",
        message="The account management portal appears to be offline.",
        ticket_type="Incident",
        queue="Technical Support",
        priority="high",
        language="en",
        source_version=51,
        tags=["Account", "Disruption", "Outage"],
    )

    assert ticket.message == "The account management portal appears to be offline."
    assert ticket.tags == ["Account", "Disruption", "Outage"]


def test_subject_is_optional_and_nullable() -> None:
    ticket = Ticket(
        message="Please clarify my invoice charges.",
        ticket_type="Request",
        queue="Billing and Payments",
        priority="medium",
        language="en",
        source_version=400,
    )

    assert ticket.subject is None
    assert ticket.tags == []


def test_missing_required_message_fails_validation() -> None:
    with pytest.raises(ValidationError):
        Ticket(
            ticket_type="Incident",
            queue="Technical Support",
            priority="high",
            language="de",
            source_version=51,
        )


def test_message_must_be_a_string() -> None:
    with pytest.raises(ValidationError):
        Ticket(
            message=123,
            ticket_type="Incident",
            queue="Technical Support",
            priority="high",
            language="en",
            source_version=52,
        )


def test_priority_must_match_observed_controlled_values() -> None:
    with pytest.raises(ValidationError):
        Ticket(
            message="The service is unavailable.",
            ticket_type="Incident",
            queue="Service Outages and Maintenance",
            priority="urgent",
            language="en",
            source_version=400,
        )


def test_extra_source_fields_are_not_part_of_the_canonical_ticket() -> None:
    with pytest.raises(ValidationError):
        Ticket(
            message="I need help with the product.",
            ticket_type="Request",
            queue="Product Support",
            priority="low",
            language="en",
            source_version=400,
            answer="This support response is not part of the canonical ticket.",
        )

