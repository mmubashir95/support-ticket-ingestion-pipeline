"""Deterministic, meaning-preserving normalization for usable ticket text.

The fixed order is Unicode NFC, HTML/entity handling, control-character
cleanup, newline normalization, and whitespace normalization. Case,
punctuation, emoji, URLs, and email addresses are deliberately preserved.
"""

import re
import unicodedata
from html.parser import HTMLParser


_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_NON_VISIBLE_TAGS = frozenset({"script", "style"})
_INLINE_HTML_TAGS = frozenset(
    {
        "a",
        "abbr",
        "area",
        "audio",
        "b",
        "base",
        "bdi",
        "bdo",
        "button",
        "canvas",
        "caption",
        "cite",
        "code",
        "col",
        "colgroup",
        "data",
        "datalist",
        "del",
        "details",
        "dfn",
        "dialog",
        "em",
        "embed",
        "fieldset",
        "form",
        "head",
        "html",
        "i",
        "iframe",
        "img",
        "input",
        "ins",
        "kbd",
        "label",
        "legend",
        "link",
        "map",
        "mark",
        "menu",
        "meta",
        "meter",
        "noscript",
        "object",
        "optgroup",
        "option",
        "output",
        "param",
        "picture",
        "progress",
        "q",
        "rp",
        "rt",
        "ruby",
        "s",
        "samp",
        "select",
        "slot",
        "small",
        "source",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "svg",
        "template",
        "textarea",
        "time",
        "title",
        "track",
        "u",
        "var",
        "video",
        "wbr",
    }
)
_HTML_TAGS = _BLOCK_TAGS | _NON_VISIBLE_TAGS | _INLINE_HTML_TAGS


class _HTMLTextExtractor(HTMLParser):
    """Extract visible HTML text while retaining block boundaries."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in _NON_VISIBLE_TAGS:
            self._hidden_depth += 1
        elif tag in _BLOCK_TAGS and self._hidden_depth == 0:
            self.parts.append("\n")
        elif tag not in _HTML_TAGS and self._hidden_depth == 0:
            self.parts.append(self.get_starttag_text())

    def handle_endtag(self, tag: str) -> None:
        if tag in _NON_VISIBLE_TAGS:
            self._hidden_depth = max(0, self._hidden_depth - 1)
        elif tag in _BLOCK_TAGS and self._hidden_depth == 0:
            self.parts.append("\n")
        elif tag not in _HTML_TAGS and self._hidden_depth == 0:
            self.parts.append(f"</{tag}>")

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag not in _HTML_TAGS and self._hidden_depth == 0:
            self.parts.append(self.get_starttag_text())
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._hidden_depth == 0:
            self.parts.append(data)


def normalize_unicode(text: str) -> str:
    """Normalize canonically equivalent Unicode text to NFC."""

    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    """Remove HTML markup, decode entities, and preserve visible text blocks."""

    parser = _HTMLTextExtractor()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def remove_control_characters(text: str) -> str:
    """Remove control characters except newlines, carriage returns, and tabs."""

    preserved_controls = {"\n", "\r", "\t"}
    return "".join(
        character
        for character in text
        if unicodedata.category(character) != "Cc"
        or character in preserved_controls
    )


def normalize_newlines(text: str) -> str:
    """Convert Windows and legacy Mac line endings to line feeds."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(text: str) -> str:
    """Normalize horizontal whitespace and keep at most one blank line."""

    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Apply the ticket text normalization policy in a fixed order."""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = remove_control_characters(text)
    text = normalize_newlines(text)
    text = normalize_whitespace(text)
    return text
