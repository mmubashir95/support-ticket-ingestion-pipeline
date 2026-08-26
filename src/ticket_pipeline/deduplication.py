"""Deterministic exact deduplication for preprocessed ticket text.

Inputs must already be normalized and PII-masked. This module deliberately
does not perform any additional text transformation or semantic comparison.
"""

import hashlib
from collections.abc import Iterable
from typing import TypedDict


class ExactDuplicateResult(TypedDict):
    """Exact-duplicate metadata for one input record."""

    record_index: int
    is_exact_duplicate: bool
    duplicate_of: int | None
    deduplication_fingerprint: str


def build_deduplication_key(subject: str | None, body: str) -> str:
    """Join a preprocessed subject and body with a fixed newline separator.

    A missing subject is represented by an empty string. Empty content is
    accepted here because usability validation belongs to an earlier stage.
    """

    canonical_subject = subject if subject is not None else ""
    return f"{canonical_subject}\n{body}"


def compute_ticket_fingerprint(subject: str | None, body: str) -> str:
    """Return a stable SHA-256 fingerprint of preprocessed ticket content."""

    deduplication_key = build_deduplication_key(subject, body)
    return hashlib.sha256(deduplication_key.encode("utf-8")).hexdigest()


def detect_exact_duplicates(
    tickets: Iterable[tuple[str | None, str]],
) -> list[ExactDuplicateResult]:
    """Classify preprocessed ``(subject, body)`` pairs in input order.

    The first occurrence of a fingerprint is canonical. Every later match
    points directly to that first record's zero-based input index.
    """

    first_index_by_fingerprint: dict[str, int] = {}
    results: list[ExactDuplicateResult] = []

    for record_index, (subject, body) in enumerate(tickets):
        fingerprint = compute_ticket_fingerprint(subject, body)
        duplicate_of = first_index_by_fingerprint.get(fingerprint)
        is_exact_duplicate = duplicate_of is not None

        if not is_exact_duplicate:
            first_index_by_fingerprint[fingerprint] = record_index

        results.append(
            {
                "record_index": record_index,
                "is_exact_duplicate": is_exact_duplicate,
                "duplicate_of": duplicate_of,
                "deduplication_fingerprint": fingerprint,
            }
        )

    return results
