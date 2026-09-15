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

The default test suite covers unit tests, integration tests, and an
end-to-end persistence test without downloading external model assets. The
real sentence-transformer semantic smoke test is opt-in because it may load or
download the embedding model:

```bash
RUN_REAL_SEMANTIC_MODEL_TEST=1 pytest tests/test_semantic_deduplication.py::test_real_sentence_transformer_smoke
```

## Implemented processing order

```text
CSV/JSON mapping -> schema validation -> message usability validation
-> normalization -> PII masking -> exact deduplication
-> semantic duplicate candidates -> language detection -> leakage checks
-> dataset manifest/version -> data-quality report -> PipelineResult
-> write_pipeline_outputs()
-> accepted.jsonl, rejected.jsonl, dataset_manifest.json, data_quality_report.json
```

Each stage is exposed as an isolated, testable module and is also wired into
the active Phase 1 Python API workflow. The repository currently does not
define a CLI entry point; run the pipeline from Python with `run_pipeline(...)`
and persist artifacts with `write_pipeline_outputs(...)`. Configuration is
code-based through `PipelineConfig`, `LanguageDetectionConfig`, and
`LeakageConfig`; there is no separate `configs/` directory.

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
probability for the leading candidate against that set, rounded to 9 decimal
places. The backend computes that probability with multithreaded
floating-point summation, so the raw value jitters in its least-significant
bits between otherwise identical runs; quantizing before the value is stored
keeps repeated runs byte-identical and the content-addressed dataset version
stable, while staying far more precise than any reported statistic. This
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

The manifest records input and processed counts. Accepted and rejected counts
are reported in the data-quality report and represented directly by the
accepted/rejected output artifacts. Helpers are provided to write/load a JSON
manifest and compare whether input, configuration, output, pipeline version,
or counts changed.

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

## Phase 1 Step 14 — Pipeline Orchestration

`src/ticket_pipeline/pipeline.py` provides the canonical in-memory Phase 1
entry point:

```python
from ticket_pipeline.pipeline import PipelineConfig, run_pipeline

result = run_pipeline(
    "data/raw/tickets_raw.csv",
    PipelineConfig(),
)
```

`PipelineConfig()` enables semantic duplicate candidates by default, which
needs the optional `[semantic]` dependency (see setup above). When it is
enabled without that dependency and without an injected `semantic_model`,
`run_pipeline` raises `PipelineExecutionError` immediately, before doing any
work, with instructions to install `[semantic]`, inject a model, or set
`PipelineConfig(semantic_deduplication_enabled=False)`.

`run_pipeline(source, config)` accepts one CSV or JSON source path supported by
the existing loaders. It returns a typed `PipelineResult`:

```text
accepted_records: list[Ticket]
rejected_records: list[RejectedRecord]
dataset_manifest: DatasetManifest
data_quality_report: DataQualityReport
```

The orchestration order is explicit:

```text
source loading
-> schema validation
-> message usability validation
-> normalization
-> PII masking
-> exact deduplication
-> semantic duplicate candidates
-> language detection
-> leakage checks
-> dataset manifest/version
-> data-quality report
```

The orchestrator composes existing stage APIs rather than reimplementing their
logic. Schema-invalid and empty-message records become `RejectedRecord`
instances with their original zero-based source index, canonical mapped record,
and existing validation failure metadata. Rejected records are not passed into
normalization, PII masking, deduplication, language detection, leakage checks,
versioning output records, or report language/leakage summaries.

`PipelineConfig` composes the existing configurable pieces: semantic
deduplication settings, optional injected semantic model, language detection
config/backend, leakage config, optional pipeline version, and optional
manifest `created_at`. Non-configurable stages such as normalization, URL/email
policy, PII masking policy, and exact deduplication continue to use their
module-owned deterministic behavior.

Dataset versioning is created once through Step 12's `create_dataset_manifest`.
The manifest uses Step 12 source fingerprinting, processing-config snapshots,
ordered accepted records, and package/passed pipeline version. The
data-quality report is then produced through Step 13's
`generate_data_quality_report`, using the stage outputs accumulated during the
same run.

Record-level quality problems are preserved as rejected records. Fatal
pipeline problems, such as an unreadable source file, invalid global
configuration, unexpected stage contract mismatch, manifest creation failure,
or report generation failure, raise `PipelineExecutionError` with the original
exception preserved as `__cause__`.

Step 14 intentionally returns an in-memory result. Step 15 persists that result
with `write_pipeline_outputs(...)`, which completes the documented Phase 1
artifact workflow.

## Phase 1 Step 15 — Accepted/Rejected Outputs

`src/ticket_pipeline/outputs.py` persists an already-created Step 14
`PipelineResult`. It does not call `run_pipeline`, rebuild the dataset
manifest, or recalculate the data-quality report.

```python
from ticket_pipeline.outputs import write_pipeline_outputs
from ticket_pipeline.pipeline import PipelineConfig, run_pipeline

result = run_pipeline("data/raw/tickets_raw.csv", PipelineConfig())
artifacts = write_pipeline_outputs(result, "data/processed/run")
```

The fixed artifact names are:

```text
accepted.jsonl
rejected.jsonl
dataset_manifest.json
data_quality_report.json
```

`accepted.jsonl` contains one final processed `Ticket` JSON object per line,
using the Pydantic JSON representation. These are the normalized, PII-masked,
language-annotated, leakage-annotated records returned by
`PipelineResult.accepted_records`.

`rejected.jsonl` contains one privacy-filtered rejected record per line. Each
object holds the source `record_index`, a sorted `present_canonical_fields`
list naming the canonical fields that carried a value, and sanitized validation
failure metadata. Because rejected records may fail before PII masking, no raw
field value from the source or the canonical mapping is written -- only field
names. Pydantic error metadata is filtered to diagnostic fields (`loc`, `msg`,
`type`, `url`); raw `input` and `ctx` values are not persisted. To inspect the
original values of a rejected row, use its `record_index` against the source
file.

`dataset_manifest.json` is the exact Step 12 manifest object from
`PipelineResult.dataset_manifest`. `data_quality_report.json` is the exact
Step 13 report object from `PipelineResult.data_quality_report`.

The output directory is created when needed. Existing Step 15 artifact files
with the fixed names are overwritten deterministically; unrelated files in the
directory are left alone. Record ordering follows the in-memory
`PipelineResult` order. Empty accepted or rejected collections produce valid
zero-byte JSONL files, while manifest and report files are still written.

Each artifact is serialized before final replacement and then written through a
temporary file in the target directory followed by atomic `os.replace`.
If writing fails, `OutputPersistenceError` is raised with the underlying
exception preserved as `__cause__`; persistence failures are not converted into
rejected ticket records. A failure after some artifacts have already been
replaced can leave those completed replacements on disk, but each individual
artifact is either the previous complete file or the new complete file.

The repository includes unit tests, integration tests, and an end-to-end
persistence test covering source input through `run_pipeline(...)`,
`write_pipeline_outputs(...)`, accepted/rejected JSONL files, the dataset
manifest, and the data-quality report. Semantic duplicate logic is covered in
the default suite with deterministic injected embeddings; the production
`sentence-transformers` backend remains covered by the opt-in smoke test shown
above. Real-model threshold calibration and duplicate-quality evaluation belong
to later model/evaluation work, not Phase 1 ingestion closure.

## Data source

Customer support ticket data source:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/tree/main
