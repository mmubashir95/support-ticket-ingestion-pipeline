"""Conservative, deterministic masking for explicit PII patterns.

This layer is intended to run after text normalization. Existing canonical
tokens are protected, and masking proceeds in the fixed order payment cards,
IPv4 addresses, phone numbers, then explicit account/customer identifiers.
"""

import re


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


def _replace_ipv4_candidate(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if all(int(octet) <= 255 for octet in candidate.split(".")):
        return "<IP_ADDRESS>"
    return candidate


def mask_ipv4_addresses(text: str) -> str:
    """Mask dotted-quad IPv4 addresses whose four octets are valid."""

    return _IPV4_CANDIDATE_PATTERN.sub(_replace_ipv4_candidate, text)


def mask_phone_numbers(text: str) -> str:
    """Mask explicitly supported international and local phone formats."""

    return _PHONE_PATTERN.sub("<PHONE>", text)


def mask_account_ids(text: str) -> str:
    """Mask IDs with an explicit ACC, ACCOUNT, CUSTOMER, or CUST prefix."""

    return _ACCOUNT_ID_PATTERN.sub("<ACCOUNT_ID>", text)


def _mask_unprotected_text(text: str) -> str:
    text = mask_payment_cards(text)
    text = mask_ipv4_addresses(text)
    text = mask_phone_numbers(text)
    return mask_account_ids(text)


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
