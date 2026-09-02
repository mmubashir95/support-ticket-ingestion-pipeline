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
| `language_detection` | Generated | `LanguageDetectionMetadata \| None` | No | Yes | GENERATED_METADATA | Result added after cleaning and deduplication; absent before that stage. |
| `leakage_check` | Generated | `LeakageCheckMetadata \| None` | No | Yes | GENERATED_METADATA | Policy warnings and future split-group information; absent before Step 11. |

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
+
normalized and PII-masked body
```

The canonical key length-prefixes each field before concatenating them
(`f"{len(subject)}:{subject}{len(body)}:{body}"`) and is encoded as UTF-8
before SHA-256 hashing. Length-prefixing is used instead of a plain
separator character because normalization preserves paragraph newlines
inside subject/body text, so a fixed separator such as `\n` could let a
value ending or starting with a newline shift the subject/body boundary and
collide with a different pair; the length prefix removes that ambiguity
regardless of what characters the fields contain. A missing subject is
represented as an empty string. Case, punctuation, emoji, and all other
supplied content remain significant; the deduplication step performs no
additional text processing. Priority, queue, language, tags, source
version, and other metadata do not participate in the exact duplicate
identity.

Detection is a single ordered pass. The first fingerprint occurrence is the
canonical record, and later matches point directly to its zero-based input
index through `duplicate_of`. Results remain aligned with input order. The
expected complexity is O(n) time and O(u) memory for `u` unique fingerprints.

Because exact deduplication occurs after PII masking, distinct raw tickets may
become exact duplicates when their only differences were masked values. Raw
PII is not recovered or compared. Empty-content acceptance remains the
responsibility of the earlier validation stage.

## Semantic Deduplication Candidates

`src/ticket_pipeline/semantic_deduplication.py` consumes exact-unique,
normalized, and PII-masked tickets. It embeds `subject + "\n" + body` with the
CPU-friendly `sentence-transformers/all-MiniLM-L6-v2` model, using batched
inference and normalized embeddings. The model is loaded once per semantic
deduplication operation. Its files are downloaded on first use unless already
present in the local Hugging Face cache.

Install the real-model dependency with:

```bash
.venv/bin/pip install -e '.[semantic]'
```

Cosine nearest-neighbor search uses scikit-learn's `NearestNeighbors` and
retrieves a configurable number of neighbors (default `top_k=5`). Self-matches
and later records are excluded. A later record is linked to the
highest-similarity qualifying earlier neighbor among those retrieved; each
pair therefore has one deterministic direction rather than both `A -> B` and
`B -> A`. Exact duplicates can be supplied through their Step 8 canonical
links and are excluded from embedding and semantic results.

The configurable default cosine threshold is `0.85`. This is a provisional
development value and has **not** been calibrated on labeled support-ticket
duplicate/non-duplicate pairs. Results expose the nearest earlier record,
similarity score, threshold, and candidate decision so the threshold can be
evaluated later. Similarity indicates a review candidate; it does not prove
that two tickets are duplicates, and this step never deletes records.

The default test suite uses injected synthetic embeddings and requires no
network access. Set `RUN_REAL_SEMANTIC_MODEL_TEST=1` to opt into the small real
model smoke test; that run may download the model. PyTorch currently has no
compatible wheel for this project's Python 3.13 macOS x86_64 environment, so
real model inference here requires a supported Python/PyTorch platform (for
example Python 3.12) while the model-independent search remains testable.

## Language Detection

`src/ticket_pipeline/language.py` runs after semantic duplicate candidate
detection and analyzes only the normalized, PII-masked customer subject and
message. When a subject exists, the stage joins it to the message with one
newline. Detection never uses the source language label, queue, type,
priority, tags, source version, answer, or other unrelated metadata.

The local `lingua-language-detector` backend is built once per reusable
`LanguageDetector` service, from a curated set of 20 languages plausible in a
global support-ticket dataset (major European, Middle Eastern, and Asian
languages), rather than Lingua's full 75-language catalog. It performs no
network calls. The generated result is nested separately from the existing
source-declared `Ticket.language` field:

```json
{
  "language": "en",
  "confidence": 0.87,
  "status": "detected"
}
```

The language is a lowercase ISO 639-1 code. Confidence is constrained to
`0.0-1.0`, and status is one of `detected`, `uncertain`, or `failed`.
`confidence` is Lingua's own native per-language probability for the leading
candidate against the curated language set, used unmodified rather than a
derived score. Restricting the candidate set (instead of using all 75
languages) keeps this native probability meaningful: spread across the full
catalog, Lingua's own confidence drops even for unambiguous text because
probability mass is diluted across many implausible candidates. The default
detection threshold is `0.80`; equality is accepted. This default is a
conservative, project-chosen starting point and, like the semantic-duplicate
cosine threshold above, has **not** been empirically calibrated against
labeled support-ticket language data. Below-threshold results retain
confidence but set language to null and status to `uncertain`.

Text containing fewer than 4 alphabetic Unicode characters is also
`uncertain`, without invoking Lingua. This protects inputs such as `OK` while
avoiding a bias against information-dense scripts: unlike Latin-script
sentences, a handful of CJK or Arabic characters can already carry enough
signal for Lingua to identify the language unambiguously, so the minimum is
kept low and the confidence threshold — which is script-agnostic — is relied
on to filter genuinely ambiguous short text. Both the threshold and minimum
are configurable. Empty, whitespace-only, and `None` inputs are handled
defensively the same way. Backend failures return `failed` metadata and are
logged without raw ticket content.

A detected non-English language does not automatically cause ticket
rejection. Language support and ticket acceptance remain separate concerns.
The detector is ticket-level only; dedicated mixed-language handling,
translation, and language-based routing are intentionally deferred. Very
short text, Roman Urdu, mixed/code-switched text, and closely related
languages remain known limitations.

## Leakage Checks

`src/ticket_pipeline/leakage.py` runs after language detection and adds
diagnostic metadata without rejecting or modifying a ticket. The governing
prediction-time policy is:

> A field is safe for a future model only if it would realistically be
> available at the intended prediction time.

The checker receives both the processed ticket fields and, when available, an
explicit source/candidate-feature mapping. This is necessary for the current
dataset because `answer` is intentionally removed during canonical mapping,
but must still be detectable as a source leakage risk. It never inspects
ticket text semantically and never logs source values.

**This mapping is optional, and that is a real limitation, not just a
convenience.** `answer` only ever reaches the checker through it, so a caller
that omits `source_fields` cannot have `answer` leakage detected at all — the
checker has no way to warn about a field it was never given. Any future
orchestration step (Step 14) must pass this mapping for the default forbidden
policy to be meaningful. `unchecked_fields` (below) exists specifically so
this omission is visible in the output rather than looking identical to a
record that was checked and found clean.

`LeakageConfig` contains configurable `forbidden_fields`, `target_fields`,
`target_proxy_fields`, and `group_fields`. The only default forbidden field is
`answer`, based directly on this dataset's post-ticket agent response column.
No target, proxy, or grouping field is configured by default. Target and proxy
checks are skipped when no target is configured; target names and proxies are
never inferred automatically. Missing configured fields are always safe and
never produce a warning.

The generated shape is:

```json
{
  "status": "warning",
  "issues": [
    {"type": "future_field", "field": "answer"}
  ],
  "targets": {
    "target_fields": ["priority"],
    "target_proxy_fields": ["assigned_team"]
  },
  "grouping": {
    "exact_duplicate_group_id": 0,
    "semantic_duplicate_group_id": null,
    "identifiers": {"conversation_id": "CONV-123"}
  },
  "unchecked_fields": []
}
```

Statuses are `clean`, `warning`, and `failed`. Only forbidden-field findings
are warnings: a populated post-outcome field is actionable evidence something
is wrong with this record at the intended prediction time. A populated target
or target-proxy field is **not** a warning and never affects status — a
labeled training row is expected to contain its own label, and its presence
is not evidence that the label leaked into a model's input features. Target
and proxy findings are reported as informational `targets` metadata instead,
naming which fields a future feature set built for that task must exclude.
This mirrors, and is deliberately consistent with, how duplicate relationships
and configured group identifiers are already treated: informational
instructions for future split grouping, not evidence that a record itself is
invalid. Exact and semantic links come directly from Steps 8 and 9; Step 11
performs no hashing, embedding, similarity search, or duplicate recomputation.

For field presence, `None`, empty/whitespace-only strings, and empty built-in
containers are empty. `False` and numeric zero are populated because they may
encode real post-outcome decisions. Group identifiers must be populated
scalar strings, integers, floats, or booleans. Customer/user identifiers can
serve as group keys but should not become model features without explicit
justification because they enable memorization.

`unchecked_fields` lists every configured forbidden/target/proxy/group field
name that was not present at all in the merged record (canonical ticket
fields plus any supplied `source_fields`) — as opposed to present but empty.
This distinguishes "this field was checked and is genuinely absent" from
"this field was never available to check," which matters most for `answer`:
if `source_fields` is omitted, `answer` always appears in `unchecked_fields`
rather than silently contributing to a `clean` status.

Leakage detection is policy-based and cannot prove a dataset is universally
leakage-free. It cannot know an unspecified prediction point, discover target
proxies automatically, infer causal timing, or detect arbitrary semantic
leakage. Duplicate and group metadata are prepared for future safe
train/validation/test splitting; splitting itself is not performed in Step 11.

## Dataset Versioning

Step 12 produces a dataset-level `DatasetManifest`; it is not copied onto
every `Ticket`. This keeps source/build provenance separate from record-level
generated metadata such as language and leakage results.

The manifest contract is:

```json
{
  "dataset_version": "ds_a91c74f22d3b",
  "created_at": "2026-09-02T11:30:00Z",
  "pipeline_version": "0.1.0",
  "source": {
    "files": [
      {
        "name": "tickets_raw.csv",
        "sha256": "<64 lowercase hex characters>",
        "size_bytes": 12345
      }
    ]
  },
  "counts": {
    "input": 10000,
    "processed": 9500
  },
  "processing_config": {
    "semantic_deduplication": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "threshold": 0.85,
      "top_k": 5,
      "batch_size": 32
    },
    "language_detection": {
      "backend": "lingua-language-detector",
      "confidence_threshold": 0.8,
      "min_alphabetic_characters": 4,
      "supported_languages": ["ENGLISH", "GERMAN", "SPANISH"]
    },
    "leakage": {
      "forbidden_fields": ["answer"],
      "target_fields": [],
      "target_proxy_fields": [],
      "group_fields": []
    }
  },
  "fingerprints": {
    "input": "<SHA-256>",
    "config": "<SHA-256>",
    "output": "<SHA-256>"
  }
}
```

The actual supported-language list contains all configured languages; the
example is shortened for readability. Accepted/rejected counts are omitted
until Step 15 can produce them reliably.

### Identity and hashing policy

All fingerprints use SHA-256. Canonical structured serialization sorts
dictionary keys, uses compact JSON separators, preserves list order, sorts
set/frozenset content by canonical value, normalizes strings to Unicode NFC,
serializes enums by value, and converts timezone-aware datetimes to UTC.
Naive datetimes, non-finite floats, non-string dictionary keys, and arbitrary
objects fail clearly.

The input fingerprint hashes the canonical, name-sorted list of source file
manifests. Each individual source hash covers exact file bytes, so source row
order and byte-level formatting are significant. Absolute paths, mtimes,
inodes, and usernames never enter identity. Stable source names are part of
identity, and duplicate names are rejected.

The output fingerprint hashes canonical processed records in ingestion order
with length-prefixed framing. Output is intentionally order-sensitive because
the current source has no stable unique ticket ID; records are not silently
sorted by non-unique content fields.

The configuration fingerprint covers settings that currently exist:
semantic-deduplication enablement/model/threshold/retrieval/batch settings,
language threshold/minimum/candidate set, and leakage field sets. Fixed
normalization, PII, and exact-deduplication policies currently have no runtime
configuration and are represented by the package `pipeline_version`.

The dataset identity hashes the three fingerprints plus `pipeline_version`
and uses the first 12 hexadecimal characters with a `ds_` prefix. `created_at`
does not affect identity. Consequently, identical content/configuration/code
versions rebuilt at different times receive the same dataset version, while a
material input, configuration, output, or pipeline-version change receives a
different identity. Manifest comparison reports each dimension separately.

The package version is the project's code-version marker and must be bumped
when releasing materially changed processing behavior. Step 12 does not add a
registry, remote artifact store, dataset retention policy, or orchestration.

The following processing remains intentionally deferred to later implementation phases:

- broader PII detection such as names and postal addresses
- accepted/rejected output generation
