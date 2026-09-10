# Support Ticket Ingestion Pipeline

A staged, local pipeline for turning raw CSV/JSON support tickets into
validated, normalized, privacy-safe, duplicate-aware records for downstream
NLP work.

## Setup and tests

The project requires Python 3.10 or newer. Install it and run the test suite
with:

```bash
python -m pip install -e '.[dev]'
pytest
```

Semantic-deduplication model inference has a separate optional dependency:

```bash
python -m pip install -e '.[semantic]'
```

## Implemented processing order

```text
CSV/JSON mapping -> schema validation -> message usability validation
-> normalization -> PII masking -> exact deduplication
-> semantic duplicate candidates -> language detection -> leakage checks
-> dataset manifest/version
```

Each stage is currently exposed as an isolated, testable module. Accepted and
rejected output orchestration belongs to a later implementation step.

## Language detection

`src/ticket_pipeline/language.py` uses the local
`lingua-language-detector` package. A `LanguageDetector` instance builds one
Lingua backend and should be reused across a batch. No remote API or model
download is involved.

Detection runs after normalization, PII masking, and duplicate analysis. It
uses only the already-processed customer subject and message, joined with a
newline when a subject is present. It never uses labels, routing fields, IDs,
versions, or other metadata, and it does not alter ticket text.

Generated metadata is separate from the source-declared `Ticket.language`
field:

```json
{
  "language_detection": {
    "language": "es",
    "confidence": 0.96,
    "status": "detected"
  }
}
```

Language codes are lowercase ISO 639-1 values. The default confidence
threshold is `0.80`. The detector is built from a curated set of 20 languages
plausible in a global support-ticket dataset, rather than Lingua's full
75-language catalog; `confidence` is Lingua's own native per-language
probability for the leading candidate against that set, used unmodified. This
default threshold is a conservative, project-chosen starting point, not a
value calibrated against labeled support-ticket data. A score below the
threshold produces `status="uncertain"` and `language=null`; the observed
score is retained. Text with fewer than 4 alphabetic characters bypasses the
backend and is uncertain with no confidence. Both settings are configurable
through `LanguageDetectionConfig`. Backend initialization or execution errors
produce `status="failed"` and are logged without ticket text.

A detected non-English language does not automatically cause ticket
rejection. The current implementation deliberately does not translate,
route, or classify tickets based on language.

Known language-identification limitations include very short messages, Roman
Urdu, code-switched or mixed-language tickets, and closely related languages.
Detection is ticket-level only; dedicated mixed-language detection is outside
the current scope.

## Leakage checks

`src/ticket_pipeline/leakage.py` performs inexpensive, policy-based checks
after language detection. Its core rule is that a future model may use a field
only when that field would realistically exist at the intended prediction
time.

`LeakageConfig` provides four explicit field sets:

- `forbidden_fields` for post-outcome or future information;
- `target_fields` for fields a future task intends to predict;
- `target_proxy_fields` for known direct proxies of configured targets;
- `group_fields` for conversation, thread, case, or customer identifiers that
  should stay together during future dataset splitting.

The current dataset's `answer` column contains a post-ticket agent response,
so it is the only default forbidden field. Targets, proxies, and group fields
default to empty because the project has not selected an ML task and its
source schema has no conversation/customer identifier. The processed ticket
does not retain `answer`; callers pass the original source/candidate feature
mapping to the checker so excluded source fields can still be audited. Because
`answer` is only ever visible through that optional mapping, a caller that
omits it cannot have `answer` leakage detected — the checker cannot warn
about a field it never received. Any future orchestration step **must** pass
this mapping for the default policy to be meaningful; see `unchecked_fields`
below for how this gap stays visible instead of looking identical to a
genuinely clean record.

Only populated forbidden fields produce warning issues. A populated target or
target-proxy field is informational, not a warning: a labeled training row is
expected to contain its own label, so target/proxy presence does not by
itself mean that field leaked into a model's input features — it means a
future feature set built for that target must exclude it. Target and proxy
findings are reported separately under `targets`. Group identifiers and the
already-computed exact/semantic duplicate links are likewise informational
grouping metadata, not warnings. They prepare records for future safe
train/validation/test splitting; Step 11 does not perform any splitting or
recompute duplicates. Customer IDs should be used as grouping keys and
excluded from model features unless their use is deliberately justified.

`None`, blank strings, whitespace-only strings, and empty lists/dictionaries
are considered empty. `False` and numeric zero are meaningful populated
values. Missing configured fields are always safe (never a warning), but a
missing field is also listed in `unchecked_fields` so a record that was never
given a chance to check a configured field (for example because
`source_fields` was omitted) is distinguishable from one that was checked and
found genuinely empty. Checks never reject a ticket and do not modify its
text or earlier processing metadata.

Leakage detection is policy-based and cannot prove a dataset is universally
leakage-free. The intended prediction point, targets, and known proxies must
be supplied by the future modeling task; automatic target/proxy discovery and
temporal or grouped splitting are outside this stage.

## Dataset versioning

`src/ticket_pipeline/versioning.py` creates a typed, local dataset manifest.
The primary identifier is content-addressed rather than timestamp-based:

```text
ds_<first 12 hex characters of identity SHA-256>
```

The identity incorporates the combined source fingerprint, processing-config
fingerprint, ordered-output fingerprint, and package pipeline version from
`pyproject.toml`. `created_at` is retained for auditability but deliberately
excluded from identity, so rebuilding identical content with identical
settings produces the same dataset version.

Source files are hashed from their exact bytes. Manifests retain only stable
file names, hashes, and byte sizes—not absolute paths, modification times, or
user/machine information. Multiple source files are sorted by file name, so
argument order does not change the combined input fingerprint. Duplicate file
names are rejected because they would be ambiguous.

Processed records are canonically serialized and hashed in ingestion order.
The current ticket contract has no stable unique record identifier, so output
fingerprints are intentionally order-sensitive. Dictionary keys and sets are
canonicalized deterministically; strings use Unicode NFC; aware datetimes are
normalized to UTC; arbitrary objects, non-finite floats, and naive datetimes
are rejected instead of being serialized with unstable `repr()` output.

The configuration snapshot contains the existing semantic-deduplication,
language-detection, and leakage settings. Fixed normalization, PII, and exact
deduplication behavior has no independent configuration today and is tracked
through the pipeline package version. When semantic deduplication is disabled,
irrelevant semantic thresholds/model settings do not affect identity.

The manifest currently records only input and processed counts. Accepted and
rejected counts are not fabricated before those outputs exist. Helpers are
provided to write/load a JSON manifest and compare whether input,
configuration, output, pipeline version, or counts changed.

## Phase 1 Step 13 — Data-quality Report

`src/ticket_pipeline/data_quality.py` adds a deterministic reporting layer over
the existing Step 1-12 outputs. It does not validate records, normalize text,
run PII detection, recompute duplicates, rerun language detection, perform
leakage checks, or create a new dataset identity. Callers pass the records and
metadata already produced by the earlier stages to
`generate_data_quality_report(...)`, which returns a typed
`DataQualityReport`.

The report contains these top-level sections:

```text
dataset_summary
validation_summary
preprocessing_summary
pii_summary
deduplication_summary
language_summary
leakage_summary
dataset_version
```

`dataset_summary` reports total input, accepted, rejected, acceptance rate, and
rejection rate. Rates use total input records as the denominator and are
rounded to six decimal places; empty datasets return `0.0` rates. The model
enforces that accepted plus rejected equals total input.

`validation_summary` aggregates the existing validation failure reasons, such
as `SCHEMA_VALIDATION_FAILED` and `EMPTY_MESSAGE`. Reason counts are record
counts. `validation_issue_events` is an event count, so a single schema-invalid
record with multiple Pydantic errors can contribute multiple issue events.

`pii_summary` aggregates count-only metadata from PII masking. The
backward-compatible `mask_pii_with_metadata(...)` helper returns masked text,
entity counts by the project's supported PII types, and a total count without
raw detected values. When subject and message are masked separately, pass both
field results as one nested record item so `records_with_pii` remains a ticket
count while `total_entities_masked` remains an entity count. Existing
normalization-produced `<EMAIL>` and `<URL>` tokens are not counted by the PII
helper because the normalization stage currently exposes no event counters for
those replacements.

`deduplication_summary` consumes Step 8 exact-duplicate results and Step 9
semantic-candidate results. Duplicate record counts are the later records that
link to an earlier record. Group counts are the number of distinct earlier
canonical records referenced by those links. Semantic similarity is not
recomputed.

`language_summary` reads `Ticket.language_detection` metadata from Step 10 and
groups detected languages plus unknown/failed outcomes. Language percentages
use the number of records with language metadata as the denominator and are
rounded to six decimal places.

`leakage_summary` reads `Ticket.leakage_check` metadata from Step 11. It
reports clean, warning, and failed record counts plus issue counts grouped by
leakage type and field name. It does not include raw leaked content.

`dataset_version` copies the authoritative Step 12 manifest identity into the
report: dataset version, pipeline version, input fingerprint, configuration
fingerprint, and output fingerprint. It deliberately does not add a new hash
or include the manifest's `created_at` timestamp in the deterministic report
identity section.

Reports can be serialized and persisted with:

```python
from ticket_pipeline.data_quality import (
    generate_data_quality_report,
    write_data_quality_report,
)

report = generate_data_quality_report(
    total_input_records=len(source_records),
    accepted_records=processed_tickets,
    dataset_manifest=manifest,
    validation_failures=validation_failures,
    exact_duplicate_results=exact_results,
    semantic_duplicate_results=semantic_results,
    pii_results=pii_results_by_record,
)
write_data_quality_report(report, "reports/data_quality_report.json")
```

Serialization uses stable sorted JSON keys, matching the manifest persistence
style. If no preprocessing or PII event metadata is supplied, the report
returns a valid section with a limitation note rather than inventing fragile
statistics from processed strings.

## Data source

Customer support ticket data source:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/tree/main
