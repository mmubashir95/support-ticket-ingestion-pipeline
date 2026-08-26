import hashlib
import re

from ticket_pipeline.deduplication import (
    build_deduplication_key,
    compute_ticket_fingerprint,
    detect_exact_duplicates,
)
from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import mask_pii


def test_fingerprint_is_deterministic_and_uses_sha256() -> None:
    subject = "Payment failed"
    body = "Cannot pay"

    fingerprint_once = compute_ticket_fingerprint(subject, body)
    fingerprint_twice = compute_ticket_fingerprint(subject, body)
    expected = hashlib.sha256(f"{subject}\n{body}".encode("utf-8")).hexdigest()

    assert fingerprint_once == fingerprint_twice == expected
    assert len(fingerprint_once) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint_once)


def test_same_subject_with_different_body_has_different_fingerprint() -> None:
    first = compute_ticket_fingerprint("Payment failed", "Cannot pay")
    second = compute_ticket_fingerprint("Payment failed", "Card rejected")

    assert first != second


def test_same_body_with_different_subject_has_different_fingerprint() -> None:
    first = compute_ticket_fingerprint("Payment failed", "Cannot pay")
    second = compute_ticket_fingerprint("Billing problem", "Cannot pay")

    assert first != second


def test_subject_body_separator_prevents_simple_boundary_ambiguity() -> None:
    first = build_deduplication_key("ab", "c")
    second = build_deduplication_key("a", "bc")

    assert first == "ab\nc"
    assert second == "a\nbc"
    assert first != second


def test_empty_and_missing_content_hashes_safely() -> None:
    assert build_deduplication_key(None, "") == "\n"
    assert compute_ticket_fingerprint(None, "") == compute_ticket_fingerprint(
        "", ""
    )


def test_second_identical_ticket_is_duplicate_of_first() -> None:
    tickets = [
        ("Payment failed", "Cannot pay"),
        ("Payment failed", "Cannot pay"),
    ]

    results = detect_exact_duplicates(tickets)

    assert results[0]["is_exact_duplicate"] is False
    assert results[0]["duplicate_of"] is None
    assert results[1]["is_exact_duplicate"] is True
    assert results[1]["duplicate_of"] == 0
    assert (
        results[0]["deduplication_fingerprint"]
        == results[1]["deduplication_fingerprint"]
    )


def test_similar_but_not_exact_text_remains_unique() -> None:
    tickets = [
        ("Login issue", "I cannot login"),
        ("Login issue", "I'm unable to sign in"),
    ]

    assert all(
        not result["is_exact_duplicate"]
        for result in detect_exact_duplicates(tickets)
    )


def test_deduplication_is_case_sensitive() -> None:
    tickets = [
        ("Payment failed", "Cannot pay"),
        ("payment failed", "Cannot pay"),
    ]

    results = detect_exact_duplicates(tickets)

    assert results[0]["deduplication_fingerprint"] != results[1][
        "deduplication_fingerprint"
    ]
    assert all(not result["is_exact_duplicate"] for result in results)


def test_deduplication_is_punctuation_sensitive() -> None:
    first = compute_ticket_fingerprint("Payment failed!", "Contact <EMAIL>")
    second = compute_ticket_fingerprint("Payment failed?", "Contact <EMAIL>")

    assert first != second


def test_unicode_content_hashes_safely() -> None:
    subject = "Payment failed 😡 — José"
    body = "München 東京"

    fingerprint = compute_ticket_fingerprint(subject, body)

    assert len(fingerprint) == 64
    assert fingerprint == compute_ticket_fingerprint(subject, body)


def test_identical_pii_masked_content_is_deduplicated() -> None:
    tickets = [
        ("Contact issue", "Call me at <PHONE>"),
        ("Contact issue", "Call me at <PHONE>"),
    ]

    results = detect_exact_duplicates(tickets)

    assert results[1]["is_exact_duplicate"] is True
    assert results[1]["duplicate_of"] == 0


def test_all_repeated_duplicates_link_to_first_occurrence() -> None:
    tickets = [("Login issue", "I cannot login")] * 3

    results = detect_exact_duplicates(tickets)

    assert [result["duplicate_of"] for result in results] == [None, 0, 0]


def test_result_order_matches_input_order() -> None:
    tickets = [
        ("A", "first"),
        ("B", "second"),
        ("A", "first"),
        ("C", "third"),
    ]

    results = detect_exact_duplicates(tickets)

    assert [result["record_index"] for result in results] == [0, 1, 2, 3]
    assert [result["duplicate_of"] for result in results] == [None, None, 0, None]


def test_distinct_tickets_are_all_unique() -> None:
    tickets = [
        ("Payment", "Cannot pay"),
        ("Login", "Cannot login"),
        ("Delivery", "Order is late"),
    ]

    results = detect_exact_duplicates(tickets)

    assert all(not result["is_exact_duplicate"] for result in results)
    assert all(result["duplicate_of"] is None for result in results)


def test_identical_technical_tickets_deduplicate_but_different_code_does_not() -> None:
    tickets = [
        ("API failure", "HTTP 500 on /api/v1/orders/123"),
        ("API failure", "HTTP 500 on /api/v1/orders/123"),
        ("API failure", "HTTP 404 on /api/v1/orders/123"),
    ]

    results = detect_exact_duplicates(tickets)

    assert [result["duplicate_of"] for result in results] == [None, 0, None]


def test_preprocessing_can_make_distinct_raw_pii_tickets_exact_duplicates() -> None:
    raw_tickets = [
        ("Payment failed!!!!!!", "Email john@example.com"),
        ("Payment failed!", "Email jane@example.com"),
    ]
    preprocessed_tickets = [
        (
            mask_pii(normalize_text(subject)),
            mask_pii(normalize_text(body)),
        )
        for subject, body in raw_tickets
    ]

    results = detect_exact_duplicates(preprocessed_tickets)

    assert preprocessed_tickets == [
        ("Payment failed!", "Email <EMAIL>"),
        ("Payment failed!", "Email <EMAIL>"),
    ]
    assert results[1]["is_exact_duplicate"] is True
    assert results[1]["duplicate_of"] == 0
