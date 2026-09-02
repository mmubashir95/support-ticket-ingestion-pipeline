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
-> semantic duplicate candidates -> language detection
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

## Data source

Customer support ticket data source:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/tree/main
