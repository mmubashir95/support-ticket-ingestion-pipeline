"""Canonical data models for support tickets.

This module defines structural schema validity only. Business checks such as
empty-message detection, HTML cleanup, PII masking, and duplicate detection are
handled by later pipeline phases.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator


Priority = Literal["low", "medium", "high"]


class LanguageDetectionStatus(str, Enum):
    """Outcome of generated ticket-language detection."""

    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    FAILED = "failed"


class LanguageDetectionMetadata(BaseModel):
    """Validated metadata produced by the language-detection stage."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: LanguageDetectionStatus

    @model_validator(mode="after")
    def validate_status_fields(self) -> "LanguageDetectionMetadata":
        """Keep language and confidence consistent with the outcome status."""

        if self.status is LanguageDetectionStatus.DETECTED:
            if self.language is None or self.confidence is None:
                raise ValueError("detected status requires language and confidence")
        elif self.language is not None:
            raise ValueError("non-detected status must not include a language")

        if (
            self.status is LanguageDetectionStatus.FAILED
            and self.confidence is not None
        ):
            raise ValueError("failed status must not include confidence")
        return self


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
    language_detection: LanguageDetectionMetadata | None = None
