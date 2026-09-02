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

## Data source

Customer support ticket data source:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/tree/main
