import pytest

from ticket_pipeline.normalization import normalize_text
from ticket_pipeline.pii import (
    mask_account_ids,
    mask_ipv4_addresses,
    mask_payment_cards,
    mask_phone_numbers,
    mask_pii,
)


@pytest.mark.parametrize(
    "phone_number",
    [
        "+1 555 123 4567",
        "+92 300 1234567",
        "0300-1234567",
        "0300 1234567",
        "(555) 123-4567",
        "555-123-4567",
    ],
)
def test_supported_phone_formats_are_masked(phone_number: str) -> None:
    assert mask_phone_numbers(phone_number) == "<PHONE>"
    assert mask_pii(phone_number) == "<PHONE>"


def test_phone_number_inside_sentence_is_masked() -> None:
    text = "Please call me at +92 300 1234567 tomorrow."

    assert mask_pii(text) == "Please call me at <PHONE> tomorrow."


@pytest.mark.parametrize(
    "technical_text",
    [
        "HTTP 404",
        "HTTP 500",
        "port 8080",
        "version 2026",
        "error 1234",
        "build 928372",
    ],
)
def test_technical_numbers_are_not_masked_as_phones(
    technical_text: str,
) -> None:
    assert mask_phone_numbers(technical_text) == technical_text
    assert mask_pii(technical_text) == technical_text


@pytest.mark.parametrize(
    "zero_leading_id",
    [
        "order 01234567890",
        "reference 09876543210",
        "tracking id 00000000001",
        "invoice 01111111111",
    ],
)
def test_zero_leading_numeric_ids_are_not_masked_as_phones(
    zero_leading_id: str,
) -> None:
    assert mask_phone_numbers(zero_leading_id) == zero_leading_id
    assert mask_pii(zero_leading_id) == zero_leading_id


@pytest.mark.parametrize(
    "ip_address",
    ["192.168.1.20", "10.0.0.1", "127.0.0.1", "8.8.8.8"],
)
def test_valid_ipv4_addresses_are_masked(ip_address: str) -> None:
    assert mask_ipv4_addresses(ip_address) == "<IP_ADDRESS>"
    assert mask_pii(ip_address) == "<IP_ADDRESS>"


@pytest.mark.parametrize(
    "invalid_address",
    ["999.999.999.999", "1.2.3", "1.2.3.4.5"],
)
def test_invalid_ipv4_shapes_are_preserved(invalid_address: str) -> None:
    assert mask_ipv4_addresses(invalid_address) == invalid_address
    assert mask_pii(invalid_address) == invalid_address


@pytest.mark.parametrize(
    "test_card_number",
    [
        "4111111111111111",
        "4111 1111 1111 1111",
        "4111-1111-1111-1111",
    ],
)
def test_luhn_valid_test_card_numbers_are_masked(
    test_card_number: str,
) -> None:
    assert mask_payment_cards(test_card_number) == "<PAYMENT_CARD>"
    assert mask_pii(test_card_number) == "<PAYMENT_CARD>"


def test_luhn_invalid_card_like_number_is_preserved() -> None:
    card_like_number = "4111 1111 1111 1112"

    assert mask_payment_cards(card_like_number) == card_like_number
    assert mask_pii(card_like_number) == card_like_number


def test_repeated_digit_sequence_is_not_treated_as_a_payment_card() -> None:
    repeated_digits = "0000 0000 0000 0000"

    assert mask_pii(repeated_digits) == repeated_digits


@pytest.mark.parametrize(
    "account_id",
    ["ACC-928372", "ACCOUNT-123456", "CUSTOMER-88291", "CUST-72882"],
)
def test_explicit_account_ids_are_masked(account_id: str) -> None:
    assert mask_account_ids(account_id) == "<ACCOUNT_ID>"
    assert mask_pii(account_id) == "<ACCOUNT_ID>"


@pytest.mark.parametrize(
    "technical_identifier",
    ["ERR-928372", "ABC-123456", "v1-928372"],
)
def test_unrecognized_identifier_prefixes_are_preserved(
    technical_identifier: str,
) -> None:
    assert mask_account_ids(technical_identifier) == technical_identifier
    assert mask_pii(technical_identifier) == technical_identifier


def test_mixed_pii_is_masked_without_changing_layout() -> None:
    text = (
        "My email is <EMAIL>, phone is +92 300 1234567,\n"
        "account is ACC-928372, and the server IP is 192.168.1.20.\n"
        "Payment card used was 4111 1111 1111 1111."
    )

    assert mask_pii(text) == (
        "My email is <EMAIL>, phone is <PHONE>,\n"
        "account is <ACCOUNT_ID>, and the server IP is <IP_ADDRESS>.\n"
        "Payment card used was <PAYMENT_CARD>."
    )


def test_technical_support_content_is_preserved() -> None:
    text = (
        "HTTP 500 on /api/v1/orders/123 using port 8080,\n"
        "build 928372, Python 3.10, version 12.4.2, error code 10061,\n"
        "ERR_CONNECTION_RESET, HTTP500, ticket #12345, iPhone 16, Windows 11."
    )

    assert mask_pii(text) == text


@pytest.mark.parametrize(
    "token",
    [
        "<EMAIL>",
        "<URL>",
        "<PHONE>",
        "<IP_ADDRESS>",
        "<PAYMENT_CARD>",
        "<ACCOUNT_ID>",
    ],
)
def test_existing_mask_tokens_remain_unchanged(token: str) -> None:
    assert mask_pii(token) == token


def test_pii_masking_is_idempotent() -> None:
    text = (
        "Contact <EMAIL> or +1 555 123 4567 about ACC-928372. "
        "Server: 192.168.1.20. Card: 4111111111111111."
    )

    masked_once = mask_pii(text)

    assert mask_pii(masked_once) == masked_once


def test_full_normalization_and_pii_sequence_is_stable() -> None:
    raw_text = (
        "<p>Contact john@example.com or +92 300 1234567!!!</p>\r\n"
        "Server: 192.168.1.20; account: ACC-928372."
    )

    processed_once = mask_pii(normalize_text(raw_text))
    processed_twice = mask_pii(normalize_text(processed_once))

    assert processed_once == (
        "Contact <EMAIL> or <PHONE>!\n\n"
        "Server: <IP_ADDRESS>; account: <ACCOUNT_ID>."
    )
    assert processed_twice == processed_once
