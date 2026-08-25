from ticket_pipeline.normalization import (
    normalize_newlines,
    normalize_text,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
    remove_html,
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


def test_punctuation_case_and_technical_text_are_preserved() -> None:
    text = "PAYMENT FAILED!!! Why?! don't E-102 v2.4.1 /api/payment C++"

    assert normalize_text(text) == text


def test_emoji_is_preserved() -> None:
    text = "Payment failed 😡 👍"

    assert normalize_text(text) == text


def test_email_and_url_are_preserved() -> None:
    text = "Contact ali@example.com or see https://example.com/payment?a=1&b=2"

    assert normalize_text(text) == text


def test_normalization_is_deterministic_and_idempotent() -> None:
    raw_text = "<p>  Cafe\u0301&nbsp;&nbsp;FAILED!!! 😡 </p>\r\n\r\n\r\nPlease   help."

    normalized_once = normalize_text(raw_text)
    normalized_twice = normalize_text(normalized_once)

    assert normalized_once == "Café FAILED!!! 😡\n\nPlease help."
    assert normalized_twice == normalized_once
