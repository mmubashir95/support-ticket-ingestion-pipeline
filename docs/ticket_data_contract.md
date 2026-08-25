# Ticket Data Contract

## Dataset Source

The contract is based on the immutable source dataset:

```text
data/source/tickets_original.csv
```

This file should remain unchanged. The inspection below was performed against the local copy of that CSV.

## Dataset Inspection Summary

- Row count: 28,587
- Column count: 16
- Source columns: `subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`, `tag_1`, `tag_2`, `tag_3`, `tag_4`, `tag_5`, `tag_6`, `tag_7`, `tag_8`

Important null findings:

- `body` has 0 blank values and is the strongest source for the required canonical `message`.
- `subject` has 3,838 blank values, so canonical `subject` is nullable.
- `answer` has 7 blank values and is excluded from the canonical `Ticket` because it is a support response.
- `tag_1` has 0 blank values, while `tag_2` through `tag_8` become progressively sparse.

Relevant categorical values:

- `type`: `Incident`, `Request`, `Problem`, `Change`
- `queue`: `Technical Support`, `Product Support`, `Customer Service`, `IT Support`, `Billing and Payments`, `Returns and Exchanges`, `Service Outages and Maintenance`, `Sales and Pre-Sales`, `Human Resources`, `General Inquiry`
- `priority`: `low`, `medium`, `high`
- `language`: `en`, `de`
- `version`: `51`, `52`, `400`

## Source Field Inventory

| Source Field | Meaning | Example | Classification | Keep? | Notes |
|---|---|---|---|---|---|
| `subject` | Ticket title or short summary | `Account Disruption` | CORE_INPUT | Yes | Customer-visible text, but blank in 3,838 rows. |
| `body` | Main customer support request text | `I am writing to report...` | CORE_INPUT | Yes | Map to canonical `message`; no blanks found. |
| `answer` | Support response to the ticket | `Thank you for reaching out...` | POTENTIAL_LEAKAGE | No | Post-ticket support text; risky as future model input. |
| `type` | Ticket category | `Incident` | METADATA | Yes | Map to `ticket_type`; useful label/metadata, not core message text. |
| `queue` | Support queue or routing group | `Technical Support` | METADATA | Yes | Retained as routing metadata. |
| `priority` | Ticket priority label | `high` | METADATA | Yes | Retained as metadata; values are controlled in this dataset. |
| `language` | Message language code | `en` | METADATA | Yes | Retained as metadata; observed values are `en` and `de`. |
| `version` | Source dataset/version marker | `400` | METADATA | Yes | Map to `source_version`; useful for traceability. |
| `tag_1` | First topic tag | `Security` | METADATA | Yes | Combined into canonical `tags`. |
| `tag_2` | Second topic tag | `Outage` | METADATA | Yes | Combined into canonical `tags`; nullable in source. |
| `tag_3` | Third topic tag | `Disruption` | METADATA | Yes | Combined into canonical `tags`; nullable in source. |
| `tag_4` | Fourth topic tag | `Data Breach` | METADATA | Yes | Combined into canonical `tags`; nullable in source. |
| `tag_5` | Fifth topic tag | `Tech Support` | METADATA | Yes | Combined into canonical `tags`; sparse. |
| `tag_6` | Sixth topic tag | `Outage` | METADATA | Yes | Combined into canonical `tags`; sparse. |
| `tag_7` | Seventh topic tag | `Campaign` | METADATA | Yes | Combined into canonical `tags`; very sparse. |
| `tag_8` | Eighth topic tag | `Maintenance` | METADATA | Yes | Combined into canonical `tags`; very sparse. |

## Field Meaning And Use

`subject` and `body` are the only source fields treated as customer-provided core input. `body` is the required main ticket text and becomes the canonical `message`.

`type`, `queue`, `priority`, `language`, `version`, and tags describe the ticket. They are retained as metadata because they may be useful for filtering, reporting, evaluation, or future supervised tasks.

`answer` is support-generated text. It is not retained in the canonical `Ticket` model because it may contain resolution information that would not be available when a new ticket arrives.

## Field Classification

| Classification | Source Fields |
|---|---|
| CORE_INPUT | `subject`, `body` |
| METADATA | `type`, `queue`, `priority`, `language`, `version`, `tag_1`, `tag_2`, `tag_3`, `tag_4`, `tag_5`, `tag_6`, `tag_7`, `tag_8` |
| POTENTIAL_LEAKAGE | `answer` |
| NOT_NEEDED | None currently. Every non-leakage source field has a clear current use in the canonical ticket contract. |

Metadata fields such as `ticket_type`, `queue`, `priority`, and `tags` can still become leakage if a future model tries to predict those exact labels. They should not automatically be used as model input features.

## Canonical Mapping

| Source Field | Canonical Field | Reason |
|---|---|---|
| `subject` | `subject` | Name is already clear. |
| `body` | `message` | `message` describes the main customer ticket text more clearly. |
| `type` | `ticket_type` | Avoids shadowing Python's built-in `type`. |
| `queue` | `queue` | Name is already clear. |
| `priority` | `priority` | Name is already clear. |
| `language` | `language` | Name is already clear. |
| `version` | `source_version` | Clarifies that the value comes from the source dataset. |
| `tag_1` through `tag_8` | `tags` | Represents source tag columns as one canonical list of tag strings. |
| `answer` | Not mapped | Excluded because it is support-generated response text. |

## Canonical Schema

| Canonical Field | Source Field | Python Type | Required? | Nullable? | Classification | Reason |
|---|---|---|---|---|---|---|
| `subject` | `subject` | `str \| None` | No | Yes | CORE_INPUT | Useful title text, but missing in many valid rows. |
| `message` | `body` | `str` | Yes | No | CORE_INPUT | Main support-ticket text; no blanks found in the dataset. |
| `ticket_type` | `type` | `str` | Yes | No | METADATA | Existing ticket category. |
| `queue` | `queue` | `str` | Yes | No | METADATA | Existing support routing metadata. |
| `priority` | `priority` | `Literal["low", "medium", "high"]` | Yes | No | METADATA | Controlled values with no blanks in the dataset. `priority` drives downstream routing/SLA behavior, so a typo'd or unexpected value has outsized operational impact; `ticket_type` and `queue` are equally controlled in this dataset but are descriptive labels that may legitimately grow as the business changes, so they are intentionally left as plain `str` rather than `Literal`. |
| `language` | `language` | `str` | Yes | No | METADATA | Language code; kept flexible because future datasets may include more languages. |
| `source_version` | `version` | `int` | Yes | No | METADATA | Source dataset marker; no blanks found. `int` (not a strict int type) intentionally accepts simple numeric strings like `"51"`, since CSV/JSON source values arrive as text; non-numeric strings like `"abc"` still fail. Covered by `tests/test_validation.py`. |
| `tags` | `tag_1` through `tag_8` | `list[str]` | No | No | METADATA | Topic labels represented as a list; absent tags should be omitted rather than stored as `None`. |

**Note:** tag aggregation is implemented in `src/ticket_pipeline/loaders.py` (`_collect_tags`, used by `map_source_record`) and covered by `tests/test_loaders.py`. Column order is preserved (`tag_1` first, `tag_8` last), blank/null/`"nan"` values are omitted, and duplicate tag values across positions are preserved as-is rather than deduplicated — 13 records in `tickets_original.csv` contain a genuine duplicate tag, and the loader keeps both occurrences.

## Potential Leakage

`answer` is a clear leakage risk because it is the support team's response after the customer ticket is created. It may contain resolution steps, confirmed causes, escalation information, or other information that would not be available at prediction time for a newly submitted ticket.

The retained metadata fields `ticket_type`, `queue`, `priority`, and `tags` are not support responses, but they are labels or routing metadata. If a future model predicts any of these fields, that same field must be excluded from model input features.

## Business Validation Deferred

The `Ticket` Pydantic model only checks structural validity: expected field names and Python types.

The following checks are intentionally deferred to later implementation phases:

- empty messages
- whitespace-only messages
- effectively empty HTML
- Unicode normalization
- HTML removal
- URL handling
- email and PII masking
- duplicate detection
- language detection
- accepted/rejected output generation

