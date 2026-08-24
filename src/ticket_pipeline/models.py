"""Canonical data models for support tickets.

This module defines structural schema validity only. Business checks such as
empty-message detection, HTML cleanup, PII masking, and duplicate detection are
handled by later pipeline phases.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr


Priority = Literal["low", "medium", "high"]


class Ticket(BaseModel):
    """Canonical representation of one support ticket."""

    model_config = ConfigDict(extra="forbid")

    subject: StrictStr | None = None
    message: StrictStr
    ticket_type: StrictStr
    queue: StrictStr
    priority: Priority
    language: StrictStr
    source_version: int
    tags: list[StrictStr] = Field(default_factory=list)
