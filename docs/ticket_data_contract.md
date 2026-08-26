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

## Schema And Business Validation Boundary

The `Ticket` Pydantic model only checks structural validity: expected field names and Python types.

`src/ticket_pipeline/validators.py` separately rejects schema-valid tickets whose
messages contain no visible content. This includes empty or whitespace-only
strings, markup-only strings, and control-character-only strings. The check is
inspection-only and does not modify the original message.

`src/ticket_pipeline/normalization.py` then applies deterministic NFC Unicode
normalization, HTML/entity handling, control-character cleanup, consistent
line endings, URL and email replacement, repeated-punctuation normalization,
and conservative whitespace cleanup. URLs beginning with `http://`, `https://`,
or `www.` become `<URL>`, and conventional email addresses become `<EMAIL>`.
Emoji, case, meaningful punctuation, and useful paragraph boundaries are
preserved. Homogeneous runs of `!`, `?`, or `.` are reduced to one character;
mixed sequences such as `?!` remain intact.

`src/ticket_pipeline/pii.py` runs after normalization and applies this
intentionally conservative masking policy:

| Data type | Action |
|---|---|
| Email | Already replaced with `<EMAIL>` during normalization |
| URL | Already replaced with `<URL>` during normalization |
| Supported phone number | Replace with `<PHONE>` |
| Valid IPv4 address | Replace with `<IP_ADDRESS>` |
| Luhn-valid payment-card-like number | Replace with `<PAYMENT_CARD>` |
| Explicit `ACC-`, `ACCOUNT-`, `CUSTOMER-`, or `CUST-` ID | Replace with `<ACCOUNT_ID>` |
| Person name | Not handled |
| Postal address | Not handled |

Existing masking tokens remain unchanged. Payment cards are checked before
phone numbers to prevent pattern collisions, IPv4 octets are validated
numerically, and account masking requires one of the listed prefixes. Ordinary
technical numbers, paths, versions, error codes, and unrecognized identifiers
are preserved. No raw detected values are logged.

**Known limitation:** a four-part version, build, or schema number whose
segments are all `0-255` (for example `10.0.19.1`) is structurally
indistinguishable from a real IPv4 address and will be masked as
`<IP_ADDRESS>`. This is an accepted tradeoff of the conservative,
regex-only design rather than a bug; resolving it would require semantic
context beyond Phase 1 scope.

## Exact Deduplication

`src/ticket_pipeline/deduplication.py` operates only on text that has already
completed normalization and PII masking. The exact duplicate identity is:

```text
normalized and PII-masked subject
\n
normalized and PII-masked body
```

The canonical key uses one fixed newline between subject and body and is
encoded as UTF-8 before SHA-256 hashing. A missing subject is represented as an
empty string. Case, punctuation, emoji, and all other supplied content remain
significant; the deduplication step performs no additional text processing.
Priority, queue, language, tags, source version, and other metadata do not
participate in the exact duplicate identity.

Detection is a single ordered pass. The first fingerprint occurrence is the
canonical record, and later matches point directly to its zero-based input
index through `duplicate_of`. Results remain aligned with input order. The
expected complexity is O(n) time and O(u) memory for `u` unique fingerprints.

Because exact deduplication occurs after PII masking, distinct raw tickets may
become exact duplicates when their only differences were masked values. Raw
PII is not recovered or compared. Empty-content acceptance remains the
responsibility of the earlier validation stage.

The following processing remains intentionally deferred to later implementation phases:

- broader PII detection such as names and postal addresses
- semantic duplicate detection
- language detection
- accepted/rejected output generation
