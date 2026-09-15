"""Classification dataset construction from trusted Phase 1 accepted records."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ticket_pipeline.models import DatasetManifest, Ticket
from ticket_pipeline.versioning import load_dataset_manifest, sha256_hex

from ticket_classification.models import ClassificationRecord


DEFAULT_TARGET_FIELD = "ticket_type"
DEFAULT_INPUT_FIELDS = ("subject", "message")
DEFAULT_RARE_CLASS_MIN_SAMPLES = 10
DEFAULT_SHORT_TEXT_MIN_WORDS = 3
PLACEHOLDER_ONLY_TOKENS = frozenset(
    {
        "<EMAIL>",
        "<URL>",
        "<PHONE>",
        "<IP_ADDRESS>",
        "<PAYMENT_CARD>",
        "<ACCOUNT_ID>",
    }
)


@dataclass(frozen=True, slots=True)
class ClassificationAuditConfig:
    """Lightweight code-based configuration for the dataset audit."""

    target_field: str = DEFAULT_TARGET_FIELD
    input_fields: tuple[str, ...] = DEFAULT_INPUT_FIELDS
    rare_class_min_samples: int = DEFAULT_RARE_CLASS_MIN_SAMPLES
    short_text_min_words: int = DEFAULT_SHORT_TEXT_MIN_WORDS
    placeholder_only_tokens: frozenset[str] = field(
        default_factory=lambda: PLACEHOLDER_ONLY_TOKENS
    )

    def __post_init__(self) -> None:
        if not isinstance(self.target_field, str) or not self.target_field.strip():
            raise ValueError("target_field must be a non-blank string")
        if (
            isinstance(self.input_fields, str)
            or not self.input_fields
            or any(not isinstance(name, str) or not name.strip() for name in self.input_fields)
        ):
            raise ValueError("input_fields must contain non-blank strings")
        if len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("input_fields must be unique")
        if self.target_field in self.input_fields:
            raise ValueError("target_field cannot also be an input field")
        if (
            isinstance(self.rare_class_min_samples, bool)
            or not isinstance(self.rare_class_min_samples, int)
            or self.rare_class_min_samples < 1
        ):
            raise ValueError("rare_class_min_samples must be a positive integer")
        if (
            isinstance(self.short_text_min_words, bool)
            or not isinstance(self.short_text_min_words, int)
            or self.short_text_min_words < 1
        ):
            raise ValueError("short_text_min_words must be a positive integer")


def load_accepted_records(path: str | Path) -> list[Ticket]:
    """Load Phase 1 ``accepted.jsonl`` records and validate them as tickets."""

    records: list[Ticket] = []
    with Path(path).open(encoding="utf-8") as accepted_file:
        for line_number, line in enumerate(accepted_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"accepted JSONL line {line_number} is not valid JSON"
                ) from error
            records.append(Ticket.model_validate(payload))
    return records


def load_optional_dataset_manifest(path: str | Path | None) -> DatasetManifest | None:
    """Load a dataset manifest when a Phase 1 manifest path is supplied."""

    if path is None:
        return None
    return load_dataset_manifest(path)


def build_classification_text(ticket: Ticket, input_fields: tuple[str, ...]) -> str:
    """Join configured text fields in a deterministic, reusable order."""

    parts: list[str] = []
    for field_name in input_fields:
        value = getattr(ticket, field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"classification input field is not text: {field_name}")
        text = value.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def target_value(ticket: Ticket, target_field: str) -> Any:
    """Return the configured target value from an accepted ticket."""

    return getattr(ticket, target_field)


# Fields that define a ticket's own content, independent of which dataset
# build produced it. ``language_detection`` and ``leakage_check`` are
# deliberately excluded: those are Phase 1 processing artifacts whose values
# (for example duplicate group ids) depend on the composition of the whole
# dataset being processed, not on this record's own content, so including
# them would make the identifier unstable across dataset regenerations even
# when the record itself did not change.
_CANONICAL_RECORD_ID_FIELDS: tuple[str, ...] = (
    "subject",
    "message",
    "ticket_type",
    "queue",
    "priority",
    "language",
    "source_version",
    "tags",
)


def _canonical_record_payload(ticket: Ticket) -> bytes:
    payload = {name: getattr(ticket, name) for name in _CANONICAL_RECORD_ID_FIELDS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")


def build_record_ids(tickets: Sequence[Ticket]) -> list[str]:
    """Build stable, content-derived classification record identifiers.

    The id is a SHA-256 digest of the record's own canonical content, so the
    same record always gets the same id regardless of its position in
    ``accepted.jsonl`` or which run produced the file. Two accepted records
    that are genuinely identical in content receive the same digest; a
    deterministic ``:<n>`` suffix disambiguates such true duplicates so every
    id returned for one call stays unique.
    """

    occurrences: dict[str, int] = {}
    record_ids: list[str] = []
    for ticket in tickets:
        digest = sha256_hex(_canonical_record_payload(ticket))
        occurrence = occurrences.get(digest, 0)
        occurrences[digest] = occurrence + 1
        record_ids.append(
            f"record:{digest}" if occurrence == 0 else f"record:{digest}:{occurrence}"
        )
    return record_ids


def build_classification_records(
    tickets: list[Ticket],
    config: ClassificationAuditConfig | None = None,
) -> list[ClassificationRecord]:
    """Build usable labeled records for future classifiers."""

    settings = config or ClassificationAuditConfig()
    record_ids = build_record_ids(tickets)
    records: list[ClassificationRecord] = []
    for ticket, record_id in zip(tickets, record_ids):
        label = target_value(ticket, settings.target_field)
        if isinstance(label, str) and label.strip():
            records.append(
                ClassificationRecord(
                    record_id=record_id,
                    text=build_classification_text(ticket, settings.input_fields),
                    label=label,
                    source_fields=list(settings.input_fields),
                )
            )
        elif isinstance(label, list) and label and all(
            isinstance(item, str) and item.strip() for item in label
        ):
            records.append(
                ClassificationRecord(
                    record_id=record_id,
                    text=build_classification_text(ticket, settings.input_fields),
                    label=list(label),
                    source_fields=list(settings.input_fields),
                )
            )
    return records

