"""Conservative, deterministic masking for explicit PII patterns.

This layer is intended to run after text normalization. Existing canonical
tokens are protected, and masking proceeds in the fixed order payment cards,
IPv4 addresses, phone numbers, then explicit account/customer identifiers.
"""

import re
from typing import TypedDict


_EXISTING_TOKEN_PATTERN = re.compile(
    r"<(?:EMAIL|URL|PHONE|IP_ADDRESS|PAYMENT_CARD|ACCOUNT_ID)>"
)
_PAYMENT_CARD_CANDIDATE_PATTERN = re.compile(
    r"(?<![0-9])[0-9](?:[ -]?[0-9]){12,}(?![0-9])"
)
_IPV4_CANDIDATE_PATTERN = re.compile(
    r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])"
)
_PHONE_PATTERN = re.compile(
    r"""
    (?<![\w])
    (?:
        \+[0-9]{1,3}(?:[ -]?[0-9]){7,12}
        | \([0-9]{3}\)[ -][0-9]{3}[ -][0-9]{4}
        | [0-9]{3}[ -][0-9]{3}[ -][0-9]{4}
        | 0[0-9]{3}[ -][0-9]{7}
    )
    (?![\w])
    """,
    flags=re.VERBOSE,
)
_ACCOUNT_ID_PATTERN = re.compile(
    r"(?<![\w])(?:ACC|ACCOUNT|CUSTOMER|CUST)-[0-9]{5,}(?![\w-])",
    flags=re.IGNORECASE,
)
SUPPORTED_PII_TYPES = (
    "payment_card",
    "ip_address",
    "phone",
    "account_id",
)


class PiiMaskingResult(TypedDict):
    """PII masking metadata for one text field, without raw detected values."""

    masked_text: str
    entities_by_type: dict[str, int]
    total_entities_masked: int


def _passes_luhn_check(digits: str) -> bool:
    """Return whether a string of ASCII digits has a valid Luhn checksum."""

    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False

    checksum = 0
    doubling_parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == doubling_parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _replace_payment_card_candidate(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = candidate.replace(" ", "").replace("-", "")
    if _passes_luhn_check(digits):
        return "<PAYMENT_CARD>"
    return candidate


def mask_payment_cards(text: str) -> str:
    """Mask 13-19 digit, Luhn-valid payment-card-like sequences."""

    return _PAYMENT_CARD_CANDIDATE_PATTERN.sub(
        _replace_payment_card_candidate, text
    )


def _mask_payment_cards_with_counts(
    text: str,
    entities_by_type: dict[str, int],
) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        digits = candidate.replace(" ", "").replace("-", "")
        if _passes_luhn_check(digits):
            entities_by_type["payment_card"] += 1
            return "<PAYMENT_CARD>"
        return candidate

    return _PAYMENT_CARD_CANDIDATE_PATTERN.sub(replace, text)


def _replace_ipv4_candidate(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if all(int(octet) <= 255 for octet in candidate.split(".")):
        return "<IP_ADDRESS>"
    return candidate


def mask_ipv4_addresses(text: str) -> str:
    """Mask dotted-quad IPv4 addresses whose four octets are valid."""

    return _IPV4_CANDIDATE_PATTERN.sub(_replace_ipv4_candidate, text)


def _mask_ipv4_addresses_with_counts(
    text: str,
    entities_by_type: dict[str, int],
) -> str:
    def replace(match: re.Match[str]) -> str:
        candidate = match.group(0)
        if all(int(octet) <= 255 for octet in candidate.split(".")):
            entities_by_type["ip_address"] += 1
            return "<IP_ADDRESS>"
        return candidate

    return _IPV4_CANDIDATE_PATTERN.sub(replace, text)


def mask_phone_numbers(text: str) -> str:
    """Mask explicitly supported international and local phone formats."""

    return _PHONE_PATTERN.sub("<PHONE>", text)


def _mask_phone_numbers_with_counts(
    text: str,
    entities_by_type: dict[str, int],
) -> str:
    def replace(match: re.Match[str]) -> str:
        entities_by_type["phone"] += 1
        return "<PHONE>"

    return _PHONE_PATTERN.sub(replace, text)


def mask_account_ids(text: str) -> str:
    """Mask IDs with an explicit ACC, ACCOUNT, CUSTOMER, or CUST prefix."""

    return _ACCOUNT_ID_PATTERN.sub("<ACCOUNT_ID>", text)


def _mask_account_ids_with_counts(
    text: str,
    entities_by_type: dict[str, int],
) -> str:
    def replace(match: re.Match[str]) -> str:
        entities_by_type["account_id"] += 1
        return "<ACCOUNT_ID>"

    return _ACCOUNT_ID_PATTERN.sub(replace, text)


def _mask_unprotected_text(text: str) -> str:
    text = mask_payment_cards(text)
    text = mask_ipv4_addresses(text)
    text = mask_phone_numbers(text)
    return mask_account_ids(text)


def _mask_unprotected_text_with_counts(
    text: str,
    entities_by_type: dict[str, int],
) -> str:
    text = _mask_payment_cards_with_counts(text, entities_by_type)
    text = _mask_ipv4_addresses_with_counts(text, entities_by_type)
    text = _mask_phone_numbers_with_counts(text, entities_by_type)
    return _mask_account_ids_with_counts(text, entities_by_type)


def mask_pii(text: str) -> str:
    """Mask supported PII while preserving existing canonical tokens."""

    masked_parts: list[str] = []
    previous_end = 0

    for token_match in _EXISTING_TOKEN_PATTERN.finditer(text):
        masked_parts.append(
            _mask_unprotected_text(text[previous_end : token_match.start()])
        )
        masked_parts.append(token_match.group(0))
        previous_end = token_match.end()

    masked_parts.append(_mask_unprotected_text(text[previous_end:]))
    return "".join(masked_parts)


def mask_pii_with_metadata(text: str) -> PiiMaskingResult:
    """Mask supported PII and return aggregate counts only.

    Existing canonical tokens such as ``<EMAIL>`` and ``<URL>`` are preserved
    and are not counted as new masking events here because they were produced
    by the earlier normalization stage.
    """

    entities_by_type = {pii_type: 0 for pii_type in SUPPORTED_PII_TYPES}
    masked_parts: list[str] = []
    previous_end = 0

    for token_match in _EXISTING_TOKEN_PATTERN.finditer(text):
        masked_parts.append(
            _mask_unprotected_text_with_counts(
                text[previous_end : token_match.start()],
                entities_by_type,
            )
        )
        masked_parts.append(token_match.group(0))
        previous_end = token_match.end()

    masked_parts.append(
        _mask_unprotected_text_with_counts(
            text[previous_end:],
            entities_by_type,
        )
    )
    masked_text = "".join(masked_parts)
    non_zero_entities = {
        pii_type: count
        for pii_type, count in entities_by_type.items()
        if count
    }
    return {
        "masked_text": masked_text,
        "entities_by_type": non_zero_entities,
        "total_entities_masked": sum(non_zero_entities.values()),
    }
