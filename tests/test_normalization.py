import pytest

from ticket_pipeline.normalization import (
    apply_text_symbol_policy,
    normalize_newlines,
    normalize_repeated_punctuation,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
    remove_html,
    replace_emails,
    replace_urls,
)


def test_unicode_normalization_makes_composed_and_decomposed_text_equal() -> None:
    composed = "café"
    decomposed = "cafe\u0301"

    assert normalize_unicode(composed) == normalize_unicode(decomposed)
    assert normalize_unicode(decomposed) == composed


def test_html_markup_is_removed_without_losing_visible_text() -> None:
    assert normalize_text("<p>Hello <b>team</b></p>") == "Hello team"


def test_block_html_does_not_concatenate_words() -> None:
    raw_text = "<p>Hello</p><p>Payment failed</p>"

    assert normalize_text(raw_text) == "Hello\n\nPayment failed"


def test_simple_malformed_html_is_handled_reasonably() -> None:
    assert normalize_text("<p>Hello <b>team</p>") == "Hello team"


def test_html_helper_decodes_entities() -> None:
    assert "Hello\xa0\xa0team" in remove_html("Hello&nbsp;&nbsp;team")


def test_html_entities_are_decoded_and_whitespace_is_normalized() -> None:
    raw_text = "Payment&nbsp;&nbsp;failed &amp; card declined &lt;today&gt;"

    assert normalize_text(raw_text) == "Payment failed & card declined <today>"


def test_non_html_semantic_placeholders_are_preserved() -> None:
    text = "Call <tel_num> about account <acc_num> or email <email>."

    assert normalize_text(text) == text


def test_script_and_style_content_is_dropped() -> None:
    raw_text = "<p>Hello</p><script>alert(1)</script><style>p{color:red}</style><p>Bye</p>"

    assert normalize_text(raw_text) == "Hello\n\nBye"


def test_stray_angle_brackets_in_technical_text_are_preserved() -> None:
    text = "5 < 10 and 10 > 5"

    assert normalize_text(text) == text


def test_leading_trailing_and_repeated_spaces_are_normalized() -> None:
    assert normalize_text("  Hello     team  ") == "Hello team"


def test_tabs_inside_text_become_single_spaces() -> None:
    assert normalize_text("Payment\t\tfailed.") == "Payment failed."


def test_line_endings_are_normalized() -> None:
    raw_text = "Hello\r\nPayment failed\rPlease help\nThank you"

    assert normalize_newlines(raw_text) == (
        "Hello\nPayment failed\nPlease help\nThank you"
    )


def test_single_newlines_and_paragraphs_are_preserved() -> None:
    raw_text = "Hi team,\nMy payment failed.\n\nPlease help."

    assert normalize_text(raw_text) == raw_text


def test_excessive_blank_lines_are_reduced_to_one() -> None:
    raw_text = "Hello\n\n\n\nPayment failed"

    assert normalize_whitespace(raw_text) == "Hello\n\nPayment failed"


def test_problematic_control_characters_are_removed() -> None:
    assert remove_control_characters("Hello\x00\x01team") == "Helloteam"
    assert normalize_text("My\x00 payment failed") == "My payment failed"


def test_normal_punctuation_case_and_technical_text_are_preserved() -> None:
    text = "PAYMENT FAILED! Why?! don't E-102 v2.4.1 /api/payment C++"

    assert normalize_text(text) == text


@pytest.mark.parametrize(
    "text",
    ["Payment failed 😡", "Thanks 👍", "Very disappointed 😢"],
)
def test_emoji_is_preserved(text: str) -> None:
    assert normalize_text(text) == text


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "www.example.com",
        "https://example.com/path?q=test",
    ],
)
def test_common_url_forms_are_replaced(url: str) -> None:
    assert replace_urls(url) == "<URL>"
    assert normalize_text(url) == "<URL>"


def test_url_inside_sentence_is_replaced_without_losing_sentence_punctuation() -> None:
    text = "Visit https://example.com/reset to continue."

    assert normalize_text(text) == "Visit <URL> to continue."


def test_url_replacement_preserves_unmatched_closing_delimiter() -> None:
    text = "See (https://example.com/reset)."

    assert normalize_text(text) == "See (<URL>)."


@pytest.mark.parametrize(
    "email",
    [
        "john@example.com",
        "john.smith@example.com",
        "john+support@example.co.uk",
    ],
)
def test_common_email_forms_are_replaced(email: str) -> None:
    assert replace_emails(email) == "<EMAIL>"
    assert normalize_text(email) == "<EMAIL>"


def test_email_inside_sentence_is_replaced() -> None:
    text = "Please contact john.smith@example.com for help."

    assert normalize_text(text) == "Please contact <EMAIL> for help."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello!!!!!!", "Hello!"),
        ("Why?????", "Why?"),
        ("Error........", "Error."),
    ],
)
def test_excessive_repeated_punctuation_is_normalized(
    text: str, expected: str
) -> None:
    assert normalize_repeated_punctuation(text) == expected
    assert normalize_text(text) == expected


def test_mixed_punctuation_sequence_is_preserved() -> None:
    assert normalize_text("Really?!") == "Really?!"


def test_mixed_support_ticket_content_follows_symbol_policy() -> None:
    text = (
        "Hi!!! My payment failed 😡. Please check "
        "https://billing.example.com/order/123 and contact me at "
        "john@example.com!!!!!"
    )

    assert normalize_text(text) == (
        "Hi! My payment failed 😡. Please check <URL> and contact me at "
        "<EMAIL>!"
    )


def test_replacement_tokens_are_stable_and_normalization_is_idempotent() -> None:
    text = "Use <URL> or contact <EMAIL>!"

    assert normalize_text(text) == text
    assert normalize_text(normalize_text(text)) == text


def test_symbol_policy_helper_applies_replacements_before_punctuation() -> None:
    text = "Email john@example.com!!! See https://example.com/reset???"

    assert apply_text_symbol_policy(text) == "Email <EMAIL>! See <URL>?"


def test_normalization_is_deterministic_and_idempotent() -> None:
    raw_text = "<p>  Cafe\u0301&nbsp;&nbsp;FAILED!!! 😡 </p>\r\n\r\n\r\nPlease   help."

    normalized_once = normalize_text(raw_text)
    normalized_twice = normalize_text(normalized_once)

    assert normalized_once == "Café FAILED! 😡\n\nPlease help."
    assert normalized_twice == normalized_once
